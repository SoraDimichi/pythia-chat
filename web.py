import logging

import gradio as gr

from config import PORT
from model import PythiaChat

logging.basicConfig(level=logging.INFO)
logging.getLogger("accelerate").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

app = PythiaChat()
app.load()

theme = gr.themes.Soft(
    primary_hue="violet",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="Pythia 12B") as demo:
    with gr.Sidebar(position="right", open=True):
        mode = gr.Radio(
            choices=["Raw", "Chat", "Trained"] if app.has_trained else ["Raw", "Chat"],
            value="Trained" if app.has_trained else "Chat",
            label="Mode",
            info="Raw = pure completion · Chat = system prompt"
            + (" · Trained = Chat + QLoRA" if app.has_trained else ""),
        )
        username = gr.Textbox(
            value="User",
            label="Name",
            info="Turn label in Chat mode",
            max_lines=1,
        )
        temperature = gr.Slider(
            minimum=0,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Temp",
            info="0 = deterministic · 1.5 = creative",
        )
        max_tokens = gr.Slider(
            minimum=32,
            maximum=1800,
            value=512,
            step=32,
            label="Length",
            info="Max output tokens (context: 2048)",
        )
        top_p = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p",
            info="Nucleus sampling — lower = focused",
        )
        top_k = gr.Slider(
            minimum=1,
            maximum=200,
            value=50,
            step=1,
            label="Top-k",
            info="Pick from top K tokens",
        )
        rep_penalty = gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.1,
            step=0.05,
            label="Repeat penalty",
            info="1.0 = off · higher = less repetition",
        )

    gr.ChatInterface(
        fn=app.generate,
        title="🔮 Pythia 12B",
        description="Local LLM · EleutherAI/pythia-12b-deduped · 4-bit quantized",
        examples=[
            ["Explain quantum computing in simple terms"],
            ["Write a short story about a robot learning to paint"],
            ["The history of artificial intelligence began in"],
            ["def fibonacci(n):"],
        ],
        additional_inputs=[mode, username, temperature, max_tokens, top_p, top_k, rep_penalty],
        concurrency_limit=1,
    )

if __name__ == "__main__":
    log.info("Starting Pythia Chat at http://localhost:%d", PORT)
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        quiet=True,
        theme=theme,
        css=".has-info { background: none !important; padding: 0 !important; font-weight: 600; }",
    )
