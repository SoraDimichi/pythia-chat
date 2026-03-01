# Pythia Chat

Web chat interface for EleutherAI's Pythia 12B language model, running locally via Docker with GPU acceleration and 4-bit quantization.

Pythia is a **base model** (not instruction-tuned), so responses are generated via text completion rather than instruction following.

## Requirements

- NVIDIA GPU with 8+ GB VRAM (tested on GTX 1080 Ti, 11GB)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

```bash
docker compose up --build -d
```

First run downloads the model (~24GB) to a persistent Docker volume. Subsequent starts load from cache in ~15 seconds.

Open http://localhost:7860

## How It Works

- **Model**: EleutherAI/pythia-12b-deduped with 4-bit quantization via bitsandbytes (~7GB VRAM)
- **Web UI**: Gradio ChatInterface with streaming responses
- **Inference**: TextIteratorStreamer for token-by-token output in a background thread
- **Turn detection**: Custom StoppingCriteria to prevent the model from generating both sides of the conversation

## Architecture

```
Docker (pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime)
├── web.py          — Gradio server + model inference
├── model-cache     — Named volume for HuggingFace model files
└── GPU passthrough — NVIDIA runtime via docker-compose
```

## Development

`web.py` is bind-mounted into the container. After editing, restart without rebuild:

```bash
docker compose restart
```

To fully rebuild (e.g. after changing Dockerfile):

```bash
docker compose up --build -d
```

## Generation Parameters

| Parameter          | Value |
|--------------------|-------|
| max_new_tokens     | 1024  |
| temperature        | 0.7   |
| top_p              | 0.9   |
| repetition_penalty | 1.1   |
| quantization       | NF4 (4-bit) |
