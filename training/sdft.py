"""
training/sdft.py — Self-Distillation Fine-Tuning stabilization pass.

Paper: Self-Distillation as Performance Recovery (2604.15794)
After tournament, train winner to stay close to yesterday's checkpoint.
Prevents catastrophic forgetting: task_loss + α * KL(student || teacher).
CKA alignment score measured before/after to verify stabilization.

Evidence: +23% accuracy recovery from forgetting, 5-fold reduction in hallucination.
"""

import os
from datetime import date
from pathlib import Path
from typing import Any

import asyncpg
import torch
import torch.nn.functional as F


async def run_sdft_stabilization(
    user_id: str,
    winner_adapter_path: str,
    pool: asyncpg.Pool,
    base_model: str = "google/gemma-4-2b-it",
    alpha: float = 0.3,
    learning_rate: float = 5e-5,
) -> dict:
    """
    Stabilize the tournament winner using yesterday's adapter as teacher.

    Args:
        winner_adapter_path: Path to tonight's winning LoRA adapter
        alpha: KL divergence weight (0.3 from paper — small but effective)
        learning_rate: Small LR — we're stabilizing, not learning new things

    Returns:
        {stable_adapter_path, cka_before, cka_after, cka_delta}
    """
    # Load yesterday's deployed adapter as teacher
    teacher_path = await _get_last_deployed_adapter(user_id, pool)
    if not teacher_path or not Path(teacher_path).exists():
        # No previous adapter — skip stabilization (first run)
        return {
            "stable_adapter_path": winner_adapter_path,
            "cka_before": None,
            "cka_after": None,
            "cka_delta": None,
            "skipped": True,
        }

    teacher_model, tokenizer = _load_peft_model(base_model, teacher_path)
    student_model, _ = _load_peft_model(base_model, winner_adapter_path)

    # Collect stabilization data: run personal holdout through teacher
    holdout_texts = await _get_holdout_texts(user_id, pool)
    if not holdout_texts:
        return {
            "stable_adapter_path": winner_adapter_path,
            "cka_before": None,
            "cka_after": None,
            "cka_delta": None,
            "skipped": True,
        }

    # Measure CKA before stabilization
    cka_before = compute_cka(teacher_model, student_model, holdout_texts, tokenizer)

    # SDFT training loop
    student_model = _sdft_train(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        holdout_texts=holdout_texts,
        alpha=alpha,
        learning_rate=learning_rate,
    )

    # Measure CKA after stabilization
    cka_after = compute_cka(teacher_model, student_model, holdout_texts, tokenizer)

    # Save stabilized adapter
    stable_path = winner_adapter_path.rstrip("/") + "_stable"
    student_model.save_pretrained(stable_path)
    tokenizer.save_pretrained(stable_path)

    result = {
        "stable_adapter_path": stable_path,
        "cka_before": cka_before,
        "cka_after": cka_after,
        "cka_delta": (cka_after - cka_before) if (cka_before and cka_after) else None,
        "skipped": False,
    }

    # Log to DB
    await _log_sdft_result(user_id, result, pool)

    return result


def _sdft_train(
    teacher_model,
    student_model,
    tokenizer,
    holdout_texts: list[str],
    alpha: float,
    learning_rate: float,
    max_steps: int = 50,
) -> Any:
    """One-epoch SDFT pass: minimize task_loss + alpha * KL(student || teacher)."""
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    dataset = Dataset.from_list([{"text": t} for t in holdout_texts])

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    loader = DataLoader(tokenized, batch_size=2, shuffle=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    step = 0
    for batch in loader:
        if step >= max_steps:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        with torch.no_grad():
            teacher_out = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_out.logits

        student_out = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        task_loss = student_out.loss
        student_logits = student_out.logits

        # KL divergence: student should stay close to teacher distribution
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / 1.0, dim=-1),
            F.softmax(teacher_logits / 1.0, dim=-1),
            reduction="batchmean",
        )

        total_loss = task_loss + alpha * kl_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    return student_model


def compute_cka(teacher_model, student_model, texts: list[str], tokenizer) -> float:
    """
    Compute Centered Kernel Alignment between teacher and student last hidden states.
    CKA ∈ [0, 1]: 1.0 = identical representation geometry, 0.0 = completely different.
    Paper uses this as diagnostic for forgetting severity.
    """
    try:
        teacher_model.eval()
        student_model.eval()

        teacher_acts = []
        student_acts = []

        with torch.no_grad():
            for text in texts[:20]:  # limit for speed
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=256
                )
                t_out = teacher_model(**inputs, output_hidden_states=True)
                s_out = student_model(**inputs, output_hidden_states=True)

                # Last hidden layer, mean-pool over sequence
                t_hidden = t_out.hidden_states[-1].mean(dim=1).squeeze()
                s_hidden = s_out.hidden_states[-1].mean(dim=1).squeeze()

                teacher_acts.append(t_hidden)
                student_acts.append(s_hidden)

        if not teacher_acts:
            return 0.0

        H_T = torch.stack(teacher_acts)  # (N, d)
        H_S = torch.stack(student_acts)  # (N, d)

        return float(_linear_cka(H_T, H_S))
    except Exception:
        return 0.0


def _linear_cka(H_X: torch.Tensor, H_Y: torch.Tensor) -> torch.Tensor:
    """Linear CKA via HSIC with centering (Kornblith et al., 2019)."""
    n = H_X.shape[0]
    centering = torch.eye(n) - torch.ones(n, n) / n

    K = H_X @ H_X.T
    L = H_Y @ H_Y.T

    K_c = centering @ K @ centering
    L_c = centering @ L @ centering

    hsic_kl = (K_c * L_c).sum()
    hsic_kk = (K_c * K_c).sum()
    hsic_ll = (L_c * L_c).sum()

    if hsic_kk <= 0 or hsic_ll <= 0:
        return torch.tensor(0.0)

    return hsic_kl / torch.sqrt(hsic_kk * hsic_ll)


def _load_peft_model(base_model: str, adapter_path: str):
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,  # Unsloth loads merged/adapter from path
            max_seq_length=2048,
            load_in_4bit=True,
        )
        return model, tokenizer
    except Exception:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, load_in_4bit=True, torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        return model, tokenizer


async def _get_last_deployed_adapter(user_id: str, pool: asyncpg.Pool) -> str | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT adapter_path FROM training_jobs
            WHERE user_id = $1 AND status = 'deployed'
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            user_id,
        )
    return row["adapter_path"] if row else None


async def _get_holdout_texts(user_id: str, pool: asyncpg.Pool, limit: int = 50) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT prompt, teacher_response FROM training_pairs
            WHERE user_id = $1 AND quality_score >= 0.8
            ORDER BY created_at DESC LIMIT $2
            """,
            user_id, limit,
        )
    return [
        f"<start_of_turn>user\n{r['prompt']}<end_of_turn>\n"
        f"<start_of_turn>model\n{r['teacher_response']}<end_of_turn>"
        for r in rows
    ]


async def _log_sdft_result(user_id: str, result: dict, pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO training_jobs (user_id, status, stage, started_at, finished_at)
            VALUES ($1, 'completed', 'sdft_stabilization', NOW(), NOW())
            """,
            user_id,
        )
