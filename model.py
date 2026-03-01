import logging
import os
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from config import MODEL_NAME, ADAPTER_PATH, MAX_CONTEXT, AI_NAME
from prompt import extract_text, build_prompt, strip_stop_string

log = logging.getLogger(__name__)


class StopOnTurnBoundary(StoppingCriteria):
    def __init__(self, stop_token_ids, prompt_length):
        self.stop_token_ids = stop_token_ids
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_length:].tolist()
        for stop_ids in self.stop_token_ids:
            if len(generated) >= len(stop_ids) and generated[-len(stop_ids):] == stop_ids:
                return True
        return False


class PythiaChat:
    def __init__(self):
        self.tokenizer = None
        self.base_model = None
        self.trained_model = None

    def load(self):
        log.info("Loading %s (4-bit quantized)...", MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa",
        )
        if ADAPTER_PATH and os.path.isdir(ADAPTER_PATH):
            try:
                from peft import PeftModel
                log.info("Loading LoRA adapter from %s", ADAPTER_PATH)
                self.trained_model = PeftModel.from_pretrained(self.base_model, ADAPTER_PATH)
                log.info("Adapter loaded.")
            except Exception as e:
                log.warning("Failed to load adapter from %s: %s", ADAPTER_PATH, e)
                log.warning("Trained mode will be unavailable.")
        vram = torch.cuda.memory_allocated() / 1024**3
        log.info("Model loaded. VRAM used: %.1f GB", vram)

    @property
    def has_trained(self):
        return self.trained_model is not None

    def _stop_strings(self, username):
        raw = [f"\n{username}:", f"\n{AI_NAME}:"]
        token_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in raw]
        clean = [s.strip() for s in raw]
        return token_ids, clean

    def _build_gen_kwargs(self, inputs, effective_max, temperature, top_p, top_k, repetition_penalty):
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=effective_max,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = int(top_k)
        else:
            gen_kwargs["do_sample"] = False
        return gen_kwargs

    def generate(self, message, history, mode, username, temperature, max_tokens, top_p, top_k, repetition_penalty):
        message = extract_text(message)
        use_stop = mode in ("Chat", "Trained")
        name = username.strip() or "User"
        stop_token_ids, stop_clean = self._stop_strings(name)

        history = history + [{"role": "user", "content": message}]
        prompt = build_prompt(history, mode, name)

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = inputs["input_ids"].shape[1]

        if prompt_length >= MAX_CONTEXT:
            yield f"[Prompt too long: {prompt_length}/{MAX_CONTEXT} tokens. Clear chat or shorten message.]"
            return

        available = MAX_CONTEXT - prompt_length
        effective_max = min(int(max_tokens), available)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = self._build_gen_kwargs(inputs, effective_max, temperature, top_p, top_k, repetition_penalty)
        gen_kwargs["streamer"] = streamer

        if use_stop:
            stop_criteria = StopOnTurnBoundary(stop_token_ids, prompt_length)
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_criteria])

        def run_generate():
            try:
                model = self.trained_model if mode == "Trained" and self.trained_model else self.base_model
                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except Exception as e:
                log.error("Generation failed: %s", e)
                streamer.text_queue.put(streamer.stop_signal)

        thread = Thread(target=run_generate)
        thread.start()

        try:
            response = ""
            for text in streamer:
                if use_stop:
                    text, stopped = strip_stop_string(response, text, stop_clean)
                    response += text
                    if stopped:
                        for _ in streamer:
                            pass
                        yield response.strip()
                        return
                else:
                    response += text
                yield response
        finally:
            thread.join()
            torch.cuda.empty_cache()
