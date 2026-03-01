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

demo = gr.ChatInterface(
    fn=app.generate,
    title="🔮 Pythia 12B",
    description="Local LLM · EleutherAI/pythia-12b-deduped · 4-bit quantized",
    examples=[
        ["Explain quantum computing in simple terms"],
        ["Write a short story about a robot learning to paint"],
        ["The history of artificial intelligence began in"],
        ["def fibonacci(n):"],
    ],
    additional_inputs=[
        gr.Radio(
            choices=["Raw", "Chat", "Trained"] if app.has_trained else ["Raw", "Chat"],
            value="Trained" if app.has_trained else "Chat",
            label="Mode",
            info="Raw = pure completion · Chat = turn labels + system prompt"
            + (" · Trained = Chat + QLoRA adapter" if app.has_trained else ""),
        ),
        gr.Textbox(
            value="User",
            label="Your name",
            info="Used in Chat mode as your turn label",
            max_lines=1,
        ),
        gr.Slider(
            minimum=0,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Temperature",
            info="0 = deterministic · 0.7 = balanced · 1.5 = creative",
        ),
        gr.Slider(
            minimum=32,
            maximum=1800,
            value=512,
            step=32,
            label="Max tokens",
            info="Output length limit (model context: 2048 total)",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p",
            info="Nucleus sampling — lower = more focused",
        ),
        gr.Slider(
            minimum=1,
            maximum=200,
            value=50,
            step=1,
            label="Top-k",
            info="Only pick from top K most probable tokens",
        ),
        gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.1,
            step=0.05,
            label="Repetition penalty",
            info="1.0 = off · higher = less repetition",
        ),
    ],
    concurrency_limit=1,
)

if __name__ == "__main__":
    log.info("Starting Pythia Chat at http://localhost:%d", PORT)
    demo.launch(server_name="0.0.0.0", server_port=PORT, quiet=True, theme=theme)
