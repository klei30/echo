"""
Standalone Unsloth training script for Gemma 4 E2B — called as a subprocess by the orchestrator.

Usage:
    python unsloth_train.py --config <path_to_json>

The JSON config has these keys:
    model_path      str  — local HF path to Gemma 4 E2B weights
    output_dir      str  — where to save the LoRA adapter
    data_path       str  — path to sft_data.json or dpo_data.json (one JSON obj per line)
    stage           str  — "sft" | "dpo"
    prev_adapter    str  — (optional) path to prev adapter to load first (NPO / DPO base)
    lora_rank       int  — default 16
    lora_alpha      int  — default 32
    epochs          int  — default 2
    batch_size      int  — default 1
    grad_accum      int  — default 8
    lr              float — default 1e-4
    max_seq_len     int  — default 2048
"""

import argparse
import json
import sys
import os

# Must be first import — Unsloth patches torch before anything else
import unsloth  # noqa: F401
from unsloth import FastModel

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_sft_dataset(rows: list[dict], tokenizer) -> Dataset:
    """Convert instruction/output pairs to chat-formatted text."""
    texts = []
    for row in rows:
        instruction = row.get("instruction", "").strip()
        response = row.get("output", "").strip()
        if not instruction or not response:
            continue
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Remove leading <bos> — Unsloth adds it during tokenisation
        if text.startswith("<bos>"):
            text = text[5:]
        texts.append({"text": text})
    return Dataset.from_list(texts)


def _build_dpo_dataset(rows: list[dict], tokenizer) -> Dataset:
    """Convert instruction/chosen/rejected triples to DPO format."""
    records = []
    for row in rows:
        prompt = row.get("instruction", "").strip()
        chosen = row.get("chosen", "").strip()
        rejected = row.get("rejected", "").strip()
        if not prompt or not chosen or not rejected:
            continue
        # DPOTrainer expects prompt / chosen / rejected as plain strings
        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return Dataset.from_list(records)


def run(cfg: dict) -> None:
    model_path = cfg["model_path"]
    output_dir = cfg["output_dir"]
    data_path = cfg["data_path"]
    stage = cfg.get("stage", "sft")
    prev_adapter = cfg.get("prev_adapter") or None
    lora_rank = int(cfg.get("lora_rank", 16))
    lora_alpha = int(cfg.get("lora_alpha", 32))
    epochs = int(cfg.get("epochs", 2))
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("grad_accum", 8))
    lr = float(cfg.get("lr", 1e-4))
    max_seq_len = int(cfg.get("max_seq_len", 2048))
    max_steps = int(cfg.get("max_steps", -1))
    logging_steps = 1 if 0 < max_steps <= 20 else 10

    print(
        f"[unsloth_train] stage={stage} pairs_file={data_path} output={output_dir} max_steps={max_steps}",
        flush=True,
    )

    # Load base model via Unsloth FastModel (handles Gemma 4 natively)
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        load_in_4bit=False,   # bf16 — same as current setup
        full_finetuning=False,
    )

    # If we have a previous adapter, load it first (NPO / DPO base)
    if prev_adapter and os.path.isdir(prev_adapter):
        print(f"[unsloth_train] Loading prev adapter from {prev_adapter}", flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, prev_adapter, is_trainable=True)
    else:
        # Fresh LoRA — target only language layers, not vision tower
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            use_rslora=True,
            random_state=3407,
        )

    rows = _load_jsonl(data_path)
    print(f"[unsloth_train] Loaded {len(rows)} pairs", flush=True)

    if stage == "sft":
        dataset = _build_sft_dataset(rows, tokenizer)
        print(f"[unsloth_train] SFT dataset size: {len(dataset)}", flush=True)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                warmup_ratio=0.1,
                num_train_epochs=epochs,
                max_steps=max_steps,
                learning_rate=lr,
                lr_scheduler_type="cosine",
                bf16=True,
                logging_steps=logging_steps,
                save_steps=200,
                output_dir=output_dir,
                optim="adamw_8bit",
                report_to="none",
                seed=3407,
            ),
        )

    elif stage == "dpo":
        dataset = _build_dpo_dataset(rows, tokenizer)
        print(f"[unsloth_train] DPO dataset size: {len(dataset)}", flush=True)

        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # implicit ref via PEFT — standard for LoRA DPO
            args=DPOConfig(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                warmup_ratio=0.1,
                num_train_epochs=1,
                max_steps=max_steps,
                learning_rate=5e-5,
                lr_scheduler_type="cosine",
                bf16=True,
                logging_steps=logging_steps,
                output_dir=output_dir,
                optim="adamw_8bit",
                report_to="none",
                beta=0.1,
                seed=3407,
                max_length=max_seq_len,
                max_prompt_length=max_seq_len // 2,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")

    print(f"[unsloth_train] Starting {stage.upper()} training...", flush=True)
    trainer.train()

    print(f"[unsloth_train] Saving adapter to {output_dir}", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[unsloth_train] Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    run(cfg)
