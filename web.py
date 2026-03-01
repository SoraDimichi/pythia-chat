import torch
import gradio as gr
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_NAME = "EleutherAI/pythia-2.8b-deduped"
STOP_STRINGS = ["\nUser:", "\nAssistant:"]
SYSTEM_PROMPT = (
    "The following is a detailed conversation between a knowledgeable assistant and a user. "
    "The assistant gives long, thorough, and informative answers.\n\n"
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
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded on GPU. VRAM used: {vram:.1f} GB")
    return model, tokenizer


def build_prompt(history):
    lines = [SYSTEM_PROMPT]
    for user_msg, bot_msg in history:
        lines.append(f"User: {user_msg}")
        if bot_msg:
            lines.append(f"Assistant: {bot_msg}")
    lines.append("Assistant:")
    return "\n".join(lines)


def chat(message, history):
    history = history + [(message, None)]
    prompt = build_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs["input_ids"].shape[1]

    stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in STOP_STRINGS]
    stop_criteria = StopOnTurnBoundary(stop_token_ids)
    stop_criteria.prompt_length = prompt_length

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
        streamer=streamer,
    )

    thread = Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    response = ""
    for text in streamer:
        for stop in STOP_STRINGS:
            stop_clean = stop.strip()
            if stop_clean in (response + text):
                text = (response + text).split(stop_clean)[0][len(response) :]
                response += text
                for _ in streamer:
                    pass
                thread.join()
                yield response.strip()
                return
        response += text
        yield response

    thread.join()


model, tokenizer = load_model()

demo = gr.ChatInterface(
    fn=chat,
    title="Pythia 2.8B Chat",
    description="Local LLM chat powered by EleutherAI/pythia-2.8b-deduped",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
