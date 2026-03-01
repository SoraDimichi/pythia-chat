import torch
import gradio as gr
from threading import Thread
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_NAME = "EleutherAI/pythia-12b-deduped"
LORA_NAME = "lamini/instruct-peft-tuned-12b"
STOP_STRINGS = ["\nUser:", "\nAssistant:"]
SYSTEM_PROMPT = (
    "Below is a conversation between User and Assistant. "
    "Assistant answers questions directly and concisely. "
    "Assistant never breaks character or writes as User.\n\n"
)


class StopOnTurnBoundary(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
        self.prompt_length = 0

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_length :].tolist()
        for stop_ids in self.stop_token_ids:
            if len(generated) >= len(stop_ids) and generated[-len(stop_ids) :] == stop_ids:
                return True
        return False


def load_model():
    print(f"Loading {MODEL_NAME} (4-bit quantized)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"Base model loaded. VRAM used: {vram:.1f} GB")

    print(f"Loading LoRA adapter: {LORA_NAME}...")
    lora_model = PeftModel.from_pretrained(base_model, LORA_NAME)
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"LoRA adapter loaded. VRAM used: {vram:.1f} GB")

    stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in STOP_STRINGS]
    stop_clean = [s.strip() for s in STOP_STRINGS]

    return base_model, lora_model, tokenizer, stop_token_ids, stop_clean


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content[0]["text"] if content else ""
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def build_prompt(history, use_lora):
    if not use_lora:
        parts = []
        for msg in history:
            text = extract_text(msg["content"])
            if msg["role"] == "user":
                parts.append(text)
        return parts[-1] if parts else ""

    lines = [SYSTEM_PROMPT]
    for msg in history:
        text = extract_text(msg["content"])
        if msg["role"] == "user":
            lines.append(f"User: {text}")
        elif msg["role"] == "assistant":
            lines.append(f"Assistant: {text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def generate(message, history, mode, temperature, max_tokens, top_p, top_k, repetition_penalty):
    message = extract_text(message)
    use_lora = mode == "LoRA Chat"
    active_model = lora_model if use_lora else base_model

    history = history + [{"role": "user", "content": message}]
    prompt = build_prompt(history, use_lora)

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        sampling = temperature > 0
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            repetition_penalty=repetition_penalty,
            do_sample=sampling,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

        if use_lora:
            stop_criteria = StopOnTurnBoundary(stop_token_ids)
            stop_criteria.prompt_length = prompt_length
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_criteria])

        def run_generate():
            try:
                active_model.generate(**gen_kwargs)
            except Exception:
                streamer.text_queue.put(streamer.stop_signal)

        thread = Thread(target=run_generate)
        thread.start()

        response = ""
        for text in streamer:
            if use_lora:
                text, stopped = strip_stop_string(response, text, stop_clean)
                response += text
                if stopped:
                    for _ in streamer:
                        pass
                    thread.join()
                    yield response.strip()
                    return
            else:
                response += text
            yield response

        thread.join()

    torch.cuda.empty_cache()


def strip_stop_string(response, text, stop_markers):
    combined = response + text
    for marker in stop_markers:
        pos = combined.find(marker)
        if pos != -1:
            return combined[:pos][len(response) :], True
    return text, False


base_model, lora_model, tokenizer, stop_token_ids, stop_clean = load_model()

demo = gr.ChatInterface(
    fn=generate,
    title="Pythia 12B",
    description="Local LLM powered by EleutherAI/pythia-12b-deduped (4-bit quantized)",
    additional_inputs=[
        gr.Radio(
            choices=["Base", "LoRA Chat"],
            value="LoRA Chat",
            label="Mode",
        ),
        gr.Slider(
            minimum=0,
            maximum=2.0,
            value=0.7,
            step=0.1,
            label="Temperature — randomness (0 = deterministic, 0.7 = balanced, 2.0 = chaos)",
        ),
        gr.Slider(
            minimum=32,
            maximum=1800,
            value=512,
            step=32,
            label="Max tokens — output length limit (model context: 2048 total)",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p — nucleus sampling, only consider tokens within this probability mass (lower = more focused)",
        ),
        gr.Slider(
            minimum=1,
            maximum=200,
            value=50,
            step=1,
            label="Top-k — only pick from top K most probable tokens (lower = more focused)",
        ),
        gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.1,
            step=0.05,
            label="Repetition penalty — penalize repeated tokens (1.0 = off, higher = less repetition)",
        ),
    ],
    concurrency_limit=1,
)

if __name__ == "__main__":
    print("Starting Pythia Chat at http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, quiet=True)
