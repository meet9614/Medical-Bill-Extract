"""
Local inference backend: base Qwen2-VL-2B + our LoRA adapter.

Exposes the same two operations the Gemini path exposes, so `benchmark.py` can
drive either behind one interface and `app/` can switch between them with an
env var.

Load once, reuse. Model load is ~20-40s; per-page inference is a fraction of a
second on an A100 and a few seconds on a T4. Reporting a latency number that
includes model load would be dishonest, so `classify`/`extract` time only the
generate call and the harness warms up first.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import ADAPTER_DIR, MODEL, PAGE_TYPES  # noqa: E402

CLASSIFY_PROMPT = (
    "Classify this document page into exactly one of the following types:\n"
    + "\n".join(f"- {c}" for c in PAGE_TYPES)
    + "\n\nRespond with the type name only, no other text."
)

EXTRACT_PROMPT = (
    "Extract every line item from this bill page as JSON.\n"
    'Schema: {"page_type": "<type>", "bill_items": '
    '[{"item_name": str, "item_amount": float, "item_rate": float|null, '
    '"item_quantity": float|null}]}\n'
    "Rules: include every row of every table; exclude subtotals and grand "
    "totals; strip currency symbols; amounts are numbers not strings.\n"
    "Respond with JSON only."
)


def _coerce_page_type(raw: str) -> str:
    """
    Snap a free-text generation onto the label set.

    A generative classifier can emit anything. Exact match first, then
    case-insensitive containment, then give up and return "Other" -- silently
    guessing a plausible class would inflate accuracy on exactly the cases where
    the model was confused.
    """
    s = raw.strip().strip(".\"'")
    for c in PAGE_TYPES:
        if s.lower() == c.lower():
            return c
    for c in PAGE_TYPES:
        if c.lower() in s.lower():
            return c
    return "Other"


def _extract_json(raw: str) -> dict:
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Recover the outermost balanced object if the model trailed off.
    start = s.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


class LocalVLM:
    def __init__(self, adapter: str | Path | None = "stage_c", device: str | None = None):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        compute_dtype = torch.bfloat16 if MODEL.use_bf16 else torch.float16

        quant_cfg = None
        if MODEL.load_in_4bit and self.device == "cuda":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        t0 = time.perf_counter()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL.model_id,
            quantization_config=quant_cfg,
            torch_dtype=compute_dtype if self.device == "cuda" else torch.float32,
            attn_implementation=MODEL.attn_implementation,
            device_map={"": 0} if self.device == "cuda" else "cpu",
        )

        self.adapter_path = None
        # "none" is the explicit base-model ablation, not a missing file.
        if adapter is not None and str(adapter).lower() != "none":
            path = Path(adapter)
            if not path.is_absolute():
                path = ADAPTER_DIR / adapter
            if path.exists():
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(path))
                model = model.merge_and_unload() if not MODEL.load_in_4bit else model
                self.adapter_path = str(path)
            else:
                print(f"warning: adapter {path} not found; running BASE model unadapted")

        model.eval()
        self.model = model
        self.processor = AutoProcessor.from_pretrained(
            MODEL.model_id, min_pixels=MODEL.min_pixels, max_pixels=MODEL.max_pixels
        )
        self.processor.tokenizer.padding_side = "left"  # correct side for generation
        self.load_seconds = round(time.perf_counter() - t0, 2)

    def _generate(self, image, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.model.device
        )

        if self.device == "cuda":
            self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy: the benchmark must be reproducible
            )
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        trimmed = out[:, inputs.input_ids.shape[1] :]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return decoded, elapsed

    def classify(self, image, max_new_tokens: int = 12) -> tuple[str, float]:
        raw, elapsed = self._generate(image, CLASSIFY_PROMPT, max_new_tokens)
        return _coerce_page_type(raw), elapsed

    def extract(self, image, max_new_tokens: int = 1536) -> tuple[dict, float]:
        raw, elapsed = self._generate(image, EXTRACT_PROMPT, max_new_tokens)
        data = _extract_json(raw)
        items = []
        for it in data.get("bill_items", []) or []:
            if not isinstance(it, dict):
                continue

            def _f(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            items.append(
                {
                    "item_name": str(it.get("item_name", "")),
                    "item_amount": _f(it.get("item_amount")) or 0.0,
                    "item_rate": _f(it.get("item_rate")),
                    "item_quantity": _f(it.get("item_quantity")),
                }
            )
        return {
            "page_type": _coerce_page_type(str(data.get("page_type", "Other"))),
            "bill_items": items,
            "raw": raw,
        }, elapsed

    def warmup(self, image, n: int = 2) -> None:
        """Burn the first few calls -- CUDA graph capture and kernel autotune
        make call #1 several times slower than steady state."""
        for _ in range(n):
            self._generate(image, CLASSIFY_PROMPT, 8)
