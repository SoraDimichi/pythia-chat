# Pythia Chat

Web chat interface for EleutherAI's Pythia 12B language model, running locally via Docker with GPU acceleration and 4-bit quantization.

## Requirements

- NVIDIA GPU with 8+ GB VRAM (tested on GTX 1080 Ti, 11GB)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

```bash
docker compose up --build -d
```

First run downloads the model (~24GB) to a persistent Docker volume. Subsequent starts load from cache in ~15 seconds.

Open http://localhost:7860

## Modes

**Base** — pure text completion. No system prompt, no formatting. The model receives your raw input and continues writing. Pythia is a base model (not instruction-tuned), so it treats everything as a document to complete.

**LoRA Chat** — dialog mode with [lamini/instruct-peft-tuned-12b](https://huggingface.co/lamini/instruct-peft-tuned-12b) adapter. Adds instruction-following behavior on top of the base model via LoRA (Low-Rank Adaptation). Uses system prompt and turn detection to maintain conversation format.

## Generation Parameters

All configurable via sliders in the UI:

| Parameter          | Range     | Default | Description                                          |
|--------------------|-----------|---------|------------------------------------------------------|
| Temperature        | 0 – 2.0  | 0.7     | Randomness (0 = deterministic, 2.0 = chaos)          |
| Max tokens         | 32 – 1800 | 512    | Output length limit (model context: 2048 total)      |
| Top-p              | 0.1 – 1.0 | 0.9   | Nucleus sampling probability mass                    |
| Top-k              | 1 – 200   | 50     | Only pick from top K most probable tokens            |
| Repetition penalty | 1.0 – 2.0 | 1.1   | Penalize repeated tokens (1.0 = off)                 |

## Architecture

```
Docker (pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime)
├── web.py          — Gradio server + model inference
├── model-cache     — Named volume for HuggingFace model files
└── GPU passthrough — NVIDIA runtime via docker-compose
```

- **Model**: EleutherAI/pythia-12b-deduped with 4-bit quantization via bitsandbytes (~7GB VRAM)
- **LoRA**: lamini/instruct-peft-tuned-12b adapter loaded via PEFT (~30MB)
- **Web UI**: Gradio ChatInterface with streaming responses
- **Inference**: TextIteratorStreamer for token-by-token output in a background thread
- **Turn detection**: Custom StoppingCriteria in LoRA mode to prevent generating both sides of the conversation

## Development

`web.py` is bind-mounted into the container. After editing, restart without rebuild:

```bash
docker compose restart
```

To fully rebuild (e.g. after changing Dockerfile):

```bash
docker compose up --build -d
```
