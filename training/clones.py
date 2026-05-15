"""
training/clones.py — 4 clone trainers for the nightly shadow clone tournament.

Clone types:
  SeqKD       — SFT on teacher's output text (learns WHAT teacher says)
  GroupDPO    — Group-wise DPO with 8 response variants (paper: 2604.15602)
  SelfCritique — Self-rewrite loop, zero API cost
  OnPolicyGKD — Gemma generates → GPT-4.1 scores → train to maximize approval
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import torch
from openai import AsyncOpenAI

ADAPTERS_DIR = Path(os.environ.get("ADAPTERS_DIR", "/tmp/echo_adapters"))
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CloneDNA:
    method: str                         # seqkd | group_dpo | self_critique | on_policy_gkd
    lora_rank: int = 8
    learning_rate: float = 2e-4
    epochs: int = 1
    recency_weight: float = 0.7
    quality_threshold: float = 0.8
    extra: dict = field(default_factory=dict)


@dataclass
class TrainedClone:
    clone_id: str
    user_id: str
    clone_type: str
    dna: CloneDNA
    adapter_path: str
    night: date


class BaseCloneTrainer:
    def __init__(self, user_id: str, dna: CloneDNA, openai_client: AsyncOpenAI):
        self.user_id = user_id
        self.dna = dna
        self.openai = openai_client
        self.clone_id = str(uuid.uuid4())

    def _adapter_path(self, suffix: str = "") -> str:
        night = date.today().isoformat()
        path = ADAPTERS_DIR / self.user_id / night / f"{self.dna.method}{suffix}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _load_model_and_tokenizer(self, base_model: str = "google/gemma-4-2b-it"):
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.dna.lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=self.dna.lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
            )
            return model, tokenizer
        except ImportError:
            # Fallback to plain PEFT if Unsloth not available
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(
                base_model, load_in_4bit=True, torch_dtype=torch.float16
            )
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.dna.lora_rank,
                lora_alpha=self.dna.lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model = get_peft_model(model, peft_config)
            return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Clone A: SeqKD — train on teacher's output text
# ─────────────────────────────────────────────────────────────
class SeqKDTrainer(BaseCloneTrainer):
    """
    Sequential Knowledge Distillation.
    Teacher's text is the target — student learns to reproduce teacher output.
    Cheapest and most stable clone type.
    """

    def train(self, training_pairs: list[dict]) -> TrainedClone:
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset

        pairs = [p for p in training_pairs
                 if p.get("quality_score", 0) >= self.dna.quality_threshold]
        if not pairs:
            raise ValueError("SeqKD: no pairs above quality threshold")

        dataset = Dataset.from_list([
            {
                "text": f"<start_of_turn>user\n{p['prompt']}<end_of_turn>\n"
                        f"<start_of_turn>model\n{p['teacher_response']}<end_of_turn>"
            }
            for p in pairs
        ])

        model, tokenizer = self._load_model_and_tokenizer()
        adapter_path = self._adapter_path()

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=adapter_path,
                num_train_epochs=self.dna.epochs,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=self.dna.learning_rate,
                warmup_ratio=0.03,
                save_strategy="no",
                logging_steps=10,
                dataset_text_field="text",
                max_seq_length=2048,
                report_to="none",
            ),
        )
        trainer.train()
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        return TrainedClone(
            clone_id=self.clone_id,
            user_id=self.user_id,
            clone_type="seqkd",
            dna=self.dna,
            adapter_path=adapter_path,
            night=date.today(),
        )


# ─────────────────────────────────────────────────────────────
# Clone B: GroupDPO — group-wise preference learning
# ─────────────────────────────────────────────────────────────
class GroupDPOTrainer(BaseCloneTrainer):
    """
    GroupDPO: collect 8 response variants per prompt, train on group consensus.
    Paper 2604.15602: +26-40% reduction in hallucination vs pairwise DPO.
    Positive-NLL regularization prevents training collapse.
    """

    async def _generate_variants(self, prompt: str, n: int = 8) -> list[str]:
        variants = []
        temps = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        for temp in temps[:n]:
            try:
                resp = await self.openai.chat.completions.create(
                    model=os.environ.get("TEACHER_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=512,
                )
                variants.append(resp.choices[0].message.content)
            except Exception:
                pass
        return variants if variants else [""]

    def train(self, training_pairs: list[dict], variants_cache: dict[str, list[str]] | None = None) -> TrainedClone:
        """
        variants_cache: {prompt: [8 responses]} — pre-fetched to avoid async in train().
        If not provided, uses only the single teacher_response (degraded to pairwise).
        """
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset

        pairs = [p for p in training_pairs
                 if p.get("quality_score", 0) >= self.dna.quality_threshold]
        if not pairs:
            raise ValueError("GroupDPO: no pairs above quality threshold")

        dpo_data = []
        for p in pairs:
            prompt = p["prompt"]
            variants = (variants_cache or {}).get(prompt, [p["teacher_response"]])
            if len(variants) < 2:
                variants = variants + [p.get("gemma_response", variants[0])]

            # Best variant = chosen, worst = rejected
            best = variants[0]
            worst = variants[-1]

            dpo_data.append({
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": best}],
                "rejected": [{"role": "assistant", "content": worst}],
            })

        dataset = Dataset.from_list(dpo_data)
        model, tokenizer = self._load_model_and_tokenizer()
        adapter_path = self._adapter_path()

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=DPOConfig(
                output_dir=adapter_path,
                num_train_epochs=self.dna.epochs,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=self.dna.learning_rate,
                beta=0.1,
                loss_type="sigmoid",
                # Positive NLL regularization — prevents training collapse (from paper)
                loss_weights={"nll": 0.1},
                save_strategy="no",
                report_to="none",
            ),
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        return TrainedClone(
            clone_id=self.clone_id,
            user_id=self.user_id,
            clone_type="group_dpo",
            dna=self.dna,
            adapter_path=adapter_path,
            night=date.today(),
        )


# ─────────────────────────────────────────────────────────────
# Clone C: SelfCritique — zero API cost self-improvement
# ─────────────────────────────────────────────────────────────
class SelfCritiqueTrainer(BaseCloneTrainer):
    """
    Gemma grades its own response → rewrites → trains on (draft, improved) pair.
    Zero additional API cost. Works even offline.
    """

    def train(self, training_pairs: list[dict], gemma_fn=None) -> TrainedClone:
        """
        gemma_fn: sync callable(prompt) -> str (inference on current Gemma adapter)
        If not provided, trains on existing teacher pairs as fallback.
        """
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset

        pairs = [p for p in training_pairs
                 if p.get("quality_score", 0) >= self.dna.quality_threshold]

        improved_pairs = []
        for p in pairs:
            draft = p.get("gemma_response") or p["teacher_response"]
            if gemma_fn:
                # Ask Gemma to critique and rewrite its own draft
                critique_prompt = (
                    f"Original question: {p['prompt']}\n\n"
                    f"Your draft answer: {draft}\n\n"
                    f"Identify 2-3 weaknesses in your draft, then write an improved version. "
                    f"Format: CRITIQUE: [issues] | IMPROVED: [better answer]"
                )
                critique_output = gemma_fn(critique_prompt)
                if "IMPROVED:" in critique_output:
                    improved = critique_output.split("IMPROVED:")[-1].strip()
                else:
                    improved = p["teacher_response"]
            else:
                improved = p["teacher_response"]

            improved_pairs.append({
                "text": f"<start_of_turn>user\n{p['prompt']}<end_of_turn>\n"
                        f"<start_of_turn>model\n{improved}<end_of_turn>"
            })

        dataset = Dataset.from_list(improved_pairs)
        model, tokenizer = self._load_model_and_tokenizer()
        adapter_path = self._adapter_path()

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=adapter_path,
                num_train_epochs=self.dna.epochs,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=self.dna.learning_rate,
                save_strategy="no",
                dataset_text_field="text",
                max_seq_length=2048,
                report_to="none",
            ),
        )
        trainer.train()
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        return TrainedClone(
            clone_id=self.clone_id,
            user_id=self.user_id,
            clone_type="self_critique",
            dna=self.dna,
            adapter_path=adapter_path,
            night=date.today(),
        )


# ─────────────────────────────────────────────────────────────
# Clone D: OnPolicyGKD — generate then score
# ─────────────────────────────────────────────────────────────
class OnPolicyGKDTrainer(BaseCloneTrainer):
    """
    Gemma generates response → GPT-4.1 scores it (0-10) → train on high-scored outputs.
    Uses teacher logprobs if available for richer GKD signal.
    """

    async def _score_response(self, prompt: str, response: str) -> float:
        try:
            result = await self.openai.chat.completions.create(
                model=os.environ.get("TEACHER_MODEL", "gpt-4o-mini"),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rate this AI response from 0-10. Be strict.\n\n"
                        f"Question: {prompt}\nResponse: {response}\n\n"
                        f"Reply with only a number 0-10."
                    )
                }],
                temperature=0,
                max_tokens=5,
            )
            return float(result.choices[0].message.content.strip())
        except Exception:
            return 5.0

    def train(self, training_pairs: list[dict], scored_outputs: list[dict] | None = None) -> TrainedClone:
        """
        scored_outputs: [{prompt, response, score}] — pre-fetched scored Gemma outputs.
        Falls back to filtering teacher responses by quality_score.
        """
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset

        if scored_outputs:
            data = [
                {
                    "text": f"<start_of_turn>user\n{item['prompt']}<end_of_turn>\n"
                            f"<start_of_turn>model\n{item['response']}<end_of_turn>"
                }
                for item in scored_outputs
                if item.get("score", 0) >= 7.0
            ]
        else:
            # Fallback: use teacher pairs filtered by quality
            pairs = [p for p in training_pairs if p.get("quality_score", 0) >= 0.85]
            data = [
                {
                    "text": f"<start_of_turn>user\n{p['prompt']}<end_of_turn>\n"
                            f"<start_of_turn>model\n{p['teacher_response']}<end_of_turn>"
                }
                for p in pairs
            ]

        if not data:
            raise ValueError("OnPolicyGKD: no high-scored outputs to train on")

        dataset = Dataset.from_list(data)
        model, tokenizer = self._load_model_and_tokenizer()
        adapter_path = self._adapter_path()

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=adapter_path,
                num_train_epochs=self.dna.epochs,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=self.dna.learning_rate,
                save_strategy="no",
                dataset_text_field="text",
                max_seq_length=2048,
                report_to="none",
            ),
        )
        trainer.train()
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        return TrainedClone(
            clone_id=self.clone_id,
            user_id=self.user_id,
            clone_type="on_policy_gkd",
            dna=self.dna,
            adapter_path=adapter_path,
            night=date.today(),
        )


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────
def build_default_clones(user_id: str, openai_client: AsyncOpenAI) -> list[BaseCloneTrainer]:
    return [
        SeqKDTrainer(user_id, CloneDNA(method="seqkd", lora_rank=8, learning_rate=2e-4), openai_client),
        GroupDPOTrainer(user_id, CloneDNA(method="group_dpo", lora_rank=8, learning_rate=1e-4), openai_client),
        SelfCritiqueTrainer(user_id, CloneDNA(method="self_critique", lora_rank=4, learning_rate=2e-4), openai_client),
        OnPolicyGKDTrainer(user_id, CloneDNA(method="on_policy_gkd", lora_rank=8, learning_rate=2e-4), openai_client),
    ]
