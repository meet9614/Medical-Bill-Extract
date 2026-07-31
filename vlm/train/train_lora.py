"""
QLoRA fine-tune of Qwen2-VL-2B-Instruct, targeting a free-tier Colab T4.

Hardware notes that are not optional on Turing:
  * fp16 only. T4 has no bf16 units; mixing bf16 compute into a fp16 graph
    fails at the first matmul with a dtype mismatch.
  * SDPA attention, not FlashAttention-2. FA2 needs sm_80+.
  * Liger kernels are incompatible with 4-bit QLoRA. Leave them off.
  * The vision tower stays frozen (config.LORA.freeze_vision_tower). Adapting it
    costs ~3GB of optimiser state for little gain -- the ViT already reads
    documents; what needs teaching is the output format.

Staged usage (each stage warm-starts from the previous adapter):

    python -m vlm.train.train_lora --data stage_a_rvlcdip_train --out stage_a --epochs 1
    python -m vlm.train.train_lora --data stage_b_cord_train   --out stage_b --init stage_a --epochs 2
    python -m vlm.train.train_lora --data stage_c_medical_train --out stage_c --init stage_b --epochs 6 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import ADAPTER_DIR, DATASET_DIR, LORA, MODEL, SEED  # noqa: E402


class ChatJsonlDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [
            json.loads(l) for l in path.read_text().splitlines() if l.strip()
        ]
        if not self.rows:
            raise SystemExit(f"{path} is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


class Collator:
    """
    Builds the batch and masks the prompt so loss is computed on the assistant
    turn only. Without this the model spends most of its gradient budget
    learning to reproduce our own instruction text, which it already knows.
    """

    def __init__(self, processor):
        self.p = processor
        self.p.tokenizer.padding_side = "right"

    def _messages(self, row: dict) -> tuple[list, list]:
        user = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}],
        }
        assistant = {
            "role": "assistant",
            "content": [{"type": "text", "text": row["target"]}],
        }
        return [user], [user, assistant]

    def __call__(self, batch: list[dict]) -> dict:
        images, full_texts, prompt_lens = [], [], []

        for row in batch:
            img = Image.open(row["image"]).convert("RGB")
            images.append(img)
            only_user, full = self._messages(row)

            full_texts.append(
                self.p.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            )
            prompt_text = self.p.apply_chat_template(
                only_user, tokenize=False, add_generation_prompt=True
            )
            # Length must be measured *after* image-token expansion, so the
            # prompt goes through the full processor with its own image.
            prompt_ids = self.p(
                text=[prompt_text], images=[img], return_tensors="pt"
            ).input_ids
            prompt_lens.append(prompt_ids.shape[1])

        inputs = self.p(
            text=full_texts, images=images, return_tensors="pt", padding=True
        )

        labels = inputs.input_ids.clone()
        labels[inputs.attention_mask == 0] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        inputs["labels"] = labels
        return inputs


def find_target_modules(model) -> list[str]:
    """
    Collect the fully-qualified names of Linear layers to adapt.

    Matching on suffix alone would also hit the vision tower's projections;
    we filter by name so the ViT stays frozen. Returning full names (rather
    than bare suffixes) makes the exclusion unambiguous to PEFT.
    """
    import bitsandbytes as bnb

    linear_types = (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        if LORA.freeze_vision_tower and (".visual." in name or name.startswith("visual.")):
            continue
        if name.split(".")[-1] in LORA.target_suffixes:
            names.append(name)
    if not names:
        raise RuntimeError("No target modules matched -- check layer naming for this model.")
    return names


def build_model(init_adapter: Path | None):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    compute_dtype = torch.bfloat16 if MODEL.use_bf16 else torch.float16

    quant_cfg = None
    if MODEL.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL.model_id,
        quantization_config=quant_cfg,
        torch_dtype=compute_dtype,
        attn_implementation=MODEL.attn_implementation,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL.model_id,
        min_pixels=MODEL.min_pixels,
        max_pixels=MODEL.max_pixels,
    )

    model.config.use_cache = False
    if MODEL.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    if init_adapter is not None:
        print(f"warm-starting from {init_adapter}")
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    else:
        targets = find_target_modules(model)
        print(f"adapting {len(targets)} linear layers (vision tower frozen: {LORA.freeze_vision_tower})")
        model = get_peft_model(
            model,
            LoraConfig(
                r=LORA.r,
                lora_alpha=LORA.alpha,
                lora_dropout=LORA.dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )

    model.print_trainable_parameters()
    return model, processor


def main() -> int:
    from transformers import Trainer, TrainingArguments, set_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset name under artifacts/datasets")
    ap.add_argument("--out", required=True, help="adapter name under artifacts/adapters")
    ap.add_argument("--init", default=None, help="adapter name to warm-start from")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    set_seed(SEED)

    data_path = DATASET_DIR / f"{args.data}.jsonl"
    if not data_path.exists():
        raise SystemExit(f"{data_path} missing. Run vlm.data.build_datasets first.")

    init = (ADAPTER_DIR / args.init) if args.init else None
    if init is not None and not init.exists():
        raise SystemExit(f"init adapter {init} not found")

    model, processor = build_model(init)
    dataset = ChatJsonlDataset(data_path)
    print(f"{len(dataset)} training examples from {data_path.name}")

    out_dir = ADAPTER_DIR / args.out
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        # T4: fp16 on, bf16 off. Inverted on Ampere+ via VLM_USE_BF16=true.
        fp16=not MODEL.use_bf16,
        bf16=MODEL.use_bf16,
        optim="paged_adamw_8bit" if MODEL.load_in_4bit else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Our collator consumes raw dict rows; Trainer must not strip them.
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=Collator(processor),
    )
    trainer.train()

    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    (out_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "base_model": MODEL.model_id,
                "data": args.data,
                "init_adapter": args.init,
                "epochs": args.epochs,
                "lr": args.lr,
                "effective_batch": args.batch_size * args.grad_accum,
                "max_pixels": MODEL.max_pixels,
                "load_in_4bit": MODEL.load_in_4bit,
                "seed": SEED,
            },
            indent=2,
        )
    )
    print(f"\nadapter -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
