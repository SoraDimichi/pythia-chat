# Pythia Chat

CLI chat application using EleutherAI's Pythia 2.8B (deduped) language model. Runs fully on GPU.

Pythia is a **base model** (not instruction-tuned), so responses are generated via text completion rather than instruction following.

## Requirements

- NVIDIA GPU with 6+ GB VRAM
- CUDA-compatible drivers
- Python 3.12 or lower (3.13 not fully supported by PyTorch)

## Setup

```bash
cd pythia-chat
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download Model

First run downloads the model (~5.6GB) to `~/.cache/huggingface/`. To pre-download:

```bash
source venv/bin/activate
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b-deduped'); AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-2.8b-deduped')"
```

## Usage

```bash
source venv/bin/activate
python chat.py
```

Type your message at the `You:` prompt. Responses stream token-by-token.

- Type `exit` or `quit` to end
- Press `Ctrl+C` to quit immediately

## Generation Parameters

| Parameter | Value |
|-----------|-------|
| max_new_tokens | 512 |
| temperature | 0.7 |
| top_p | 0.9 |
| repetition_penalty | 1.1 |
