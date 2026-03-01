import torch
import gradio as gr
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_NAME = "EleutherAI/pythia-12b-deduped"
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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded on GPU. VRAM used: {vram:.1f} GB")

    stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in STOP_STRINGS]
    stop_clean = [s.strip() for s in STOP_STRINGS]

    return model, tokenizer, stop_token_ids, stop_clean


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content[0]["text"] if content else ""
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def build_prompt(history):
    lines = [SYSTEM_PROMPT]
    for msg in history:
        text = extract_text(msg["content"])
        if msg["role"] == "user":
            lines.append(f"User: {text}")
        elif msg["role"] == "assistant":
            lines.append(f"Assistant: {text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def strip_stop_string(response, text, stop_markers):
    combined = response + text
    for marker in stop_markers:
        pos = combined.find(marker)
        if pos != -1:
            return combined[:pos][len(response) :], True
    return text, False


def chat(message, history):
    message = extract_text(message)
    history = history + [{"role": "user", "content": message}]
    prompt = build_prompt(history)

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = inputs["input_ids"].shape[1]

        stop_criteria = StopOnTurnBoundary(stop_token_ids)
        stop_criteria.prompt_length = prompt_length

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stop_criteria]),
            streamer=streamer,
        )

        def run_generate():
            try:
                model.generate(**gen_kwargs)
            except Exception:
                streamer.text_queue.put(streamer.stop_signal)

        thread = Thread(target=run_generate)
        thread.start()

        response = ""
        for text in streamer:
            text, stopped = strip_stop_string(response, text, stop_clean)
            response += text
            if stopped:
                for _ in streamer:
                    pass
                thread.join()
                yield response.strip()
                return
            yield response

        thread.join()

    torch.cuda.empty_cache()


model, tokenizer, stop_token_ids, stop_clean = load_model()

demo = gr.ChatInterface(
    fn=chat,
    title="Pythia 12B Chat",
    description="Local LLM chat powered by EleutherAI/pythia-12b-deduped (4-bit quantized)",
    concurrency_limit=1,
)

if __name__ == "__main__":
    print("Starting Pythia Chat at http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, quiet=True)
