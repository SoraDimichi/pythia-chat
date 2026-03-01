# Pythia Chat

Local chat UI for EleutherAI's Pythia 12B with 4-bit quantization and optional QLoRA adapter.

## Requirements

- NVIDIA GPU with 8+ GB VRAM
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

```bash
docker compose up --build
```

First run downloads the model (~24GB) to a persistent Docker volume. Open http://localhost:7860

## Training

QLoRA fine-tuning on [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned):

```bash
docker compose run --build train
```

Options:

```bash
docker compose run --build train python train.py --epochs 2 --lora-rank 8    # custom params
docker compose run --build train python train.py --max-samples 5             # quick test
docker compose run --build train python train.py --output-dir ./.adapters/v2  # custom output
```

Adapter saves to `./.adapters/pythia-chat-v1` by default. To use it, set in `config.py`:

```python
ADAPTER_PATH = "./.adapters/pythia-chat-v1"
```

## Modes

- **Raw** — pure text completion, no formatting
- **Chat** — turn-based dialog with system prompt (base model)
- **Trained** — Chat + QLoRA adapter (appears when adapter is loaded)

## Development

After editing source files, rebuild:

```bash
docker compose up --build
```
