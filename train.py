"""QLoRA fine-tuning script for Pythia-12B chat adapter.

Trains a LoRA adapter on yahma/alpaca-cleaned dataset using the same
prompt format as the Chat mode in prompt.py.

Usage:
    python train.py                          # full training
    python train.py --max-samples 5          # quick test run
    python train.py --epochs 2 --lora-rank 8 # custom params
"""

import argparse
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from config import MODEL_NAME, AI_NAME
from prompt import system_prefix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

USERNAME = "User"
SYSTEM_PREFIX = system_prefix(USERNAME)


def format_alpaca(example):
    """Convert alpaca instruction/input/output to Chat prompt format."""
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    if inp:
        user_message = f"{instruction}\n{inp}"
    else:
        user_message = instruction

    return {
        "text": (
            f"{SYSTEM_PREFIX}"
            f"{USERNAME}: {user_message}\n"
            f"{AI_NAME}: {output}"
        )
    }


def verify_collator(tokenizer, collator, dataset, num_samples=3):
    """Verify that DataCollatorForCompletionOnlyLM finds the response template."""
    issues = 0
    check_count = min(num_samples, len(dataset))

    for i in range(check_count):
        text = dataset[i]["text"]
        tokenized = tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"][0].tolist()

        response_token_ids = collator.response_token_ids
        found = False
        for j in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[j:j + len(response_token_ids)] == response_token_ids:
                found = True
                break

        if not found:
            log.warning("Sample %d: response template NOT found in tokens!", i)
            log.warning("  Text: %s...", text[:120])
            issues += 1
        elif i < 2:
            marker_pos = text.find(f"{AI_NAME}:")
            log.info("Sample %d OK. Prompt: %s...", i, text[:80])
            log.info("  Response starts at: '%s...'", text[marker_pos:marker_pos + 60])

    if issues > 0:
        log.error("%d/%d samples have broken response template! Training will skip them.", issues, check_count)
    else:
        log.info("All %d checked samples have valid response template.", check_count)

    return issues


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Pythia-12B")
    parser.add_argument(
        "--output-dir",
        default="./.adapters/pythia-chat-v1",
        help="Directory to save the trained adapter (default: ./.adapters/pythia-chat-v1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size for test runs (default: use all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=== QLoRA Training ===")
    log.info("Model: %s", MODEL_NAME)
    log.info("Output: %s", args.output_dir)
    log.info("Epochs: %d, LoRA rank: %d", args.epochs, args.lora_rank)
    if args.max_samples:
        log.info("Max samples: %d (test run)", args.max_samples)

    # --- Tokenizer ---
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Model (4-bit quantized) ---
    log.info("Loading model in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    vram = torch.cuda.memory_allocated() / 1024**3
    log.info("Base model loaded. VRAM: %.1f GB", vram)

    # --- LoRA config (passed to SFTTrainer, not applied manually) ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )

    # --- Dataset ---
    log.info("Loading yahma/alpaca-cleaned dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    log.info("Formatting %d examples...", len(dataset))
    dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)

    # --- Train/eval split (95/5) ---
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    log.info("Split: %d train, %d eval", len(train_dataset), len(eval_dataset))

    # --- Data collator (train only on response tokens) ---
    # Use token IDs to avoid tokenization mismatch
    response_marker = f"\n{AI_NAME}:"
    response_token_ids = tokenizer.encode(response_marker, add_special_tokens=False)
    log.info("Response template: %r -> token IDs: %s", response_marker, response_token_ids)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_token_ids,
        tokenizer=tokenizer,
    )

    # --- Verify collator finds the template ---
    issues = verify_collator(tokenizer, collator, train_dataset)
    if issues > 0 and not args.max_samples:
        log.error("Aborting: response template issues detected. Fix format before full training.")
        return

    # --- Training config ---
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        max_seq_length=2048,
        dataset_text_field="text",
        report_to="none",
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    log.info("Starting training...")
    trainer.train()

    # --- Save ---
    log.info("Saving adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    vram = torch.cuda.memory_allocated() / 1024**3
    log.info("Done! Final VRAM: %.1f GB", vram)
    log.info("Adapter saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
