import time
import torch
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


STOP_STRINGS = ["\nUser:", "\nAssistant:"]


class StopOnTurnBoundary(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
        self.prompt_length = 0

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_length:].tolist()
        for stop_ids in self.stop_token_ids:
            if len(generated) >= len(stop_ids) and generated[-len(stop_ids):] == stop_ids:
                return True
        return False


def load_model():
    model_name = "EleutherAI/pythia-2.8b-deduped"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    print(f"Model loaded on GPU.")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    print()

    return model, tokenizer


SYSTEM_PROMPT = (
    "The following is a detailed conversation between a knowledgeable assistant and a user. "
    "The assistant gives long, thorough, and informative answers.\n\n"
)


def build_prompt(history):
    lines = [SYSTEM_PROMPT]
    for role, text in history:
        if role == "user":
            lines.append(f"User: {text}")
        else:
            lines.append(f"Assistant: {text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def generate_response(model, tokenizer, history):
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

    def run_generate():
        try:
            model.generate(**gen_kwargs)
        except Exception as e:
            streamer.text_queue.put(e)

    thread = Thread(target=run_generate)
    thread.start()

    print("Pythia: ", end="", flush=True)
    start = time.time()
    response = ""

    for text in streamer:
        # Check for stop strings before printing
        for stop in STOP_STRINGS:
            stop_clean = stop.strip()
            if stop_clean in (response + text):
                text = (response + text).split(stop_clean)[0]
                text = text[len(response):]
                print(text, end="", flush=True)
                response += text
                # Drain remaining tokens so thread can finish
                for _ in streamer:
                    pass
                thread.join()
                elapsed = time.time() - start
                print()
                return response.strip()

        print(text, end="", flush=True)
        response += text

    thread.join()
    print()
    return response.strip()


def main():
    model, tokenizer = load_model()

    print("Pythia 2.8B Chat")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        history.append(("user", user_input))
        response = generate_response(model, tokenizer, history)
        history.append(("assistant", response))


if __name__ == "__main__":
    main()
