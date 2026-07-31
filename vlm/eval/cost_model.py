"""
Cost model for the API-vs-owned-weights comparison.

The honest framing, because this is where these comparisons usually go wrong:

  * Gemini's cost is purely marginal -- $0 fixed, $X per page, forever.
  * A fine-tuned local model is the opposite -- a fixed training cost, then a
    marginal cost that depends entirely on where it runs. On rented GPU it is
    (hourly rate / throughput). On the user's phone, the marginal cost to you is
    genuinely zero, and the real cost moved to engineering and distribution.

  So "4x cheaper" is not a property of the model. It is a property of the model
  AND a volume assumption AND a deployment target. This module makes you state
  all three, and `breakeven_pages` tells you the volume below which the API is
  simply the cheaper choice -- which for most pilots it is.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import (  # noqa: E402
    GEMINI_INPUT_USD_PER_MTOK,
    GEMINI_OUTPUT_USD_PER_MTOK,
    GEMINI_PRICE_AS_OF,
    GPU_HOURLY_USD,
)


@dataclass
class ApiCost:
    pages: int
    input_tokens: int
    output_tokens: int
    input_usd: float
    output_usd: float
    total_usd: float
    usd_per_1000_pages: float
    price_as_of: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LocalCost:
    pages: int
    deployment: str
    seconds_per_page: float
    gpu_hourly_usd: float
    marginal_usd_per_1000_pages: float
    training_usd: float
    total_usd_at_volume: float
    usd_per_1000_pages_amortised: float

    def to_dict(self) -> dict:
        return asdict(self)


def api_cost(pages: int, input_tokens: int, output_tokens: int) -> ApiCost:
    in_usd = input_tokens / 1e6 * GEMINI_INPUT_USD_PER_MTOK
    out_usd = output_tokens / 1e6 * GEMINI_OUTPUT_USD_PER_MTOK
    total = in_usd + out_usd
    return ApiCost(
        pages=pages,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_usd=round(in_usd, 6),
        output_usd=round(out_usd, 6),
        total_usd=round(total, 6),
        usd_per_1000_pages=round(total / pages * 1000, 4) if pages else 0.0,
        price_as_of=GEMINI_PRICE_AS_OF,
    )


def local_cost(
    pages: int,
    seconds_per_page: float,
    deployment: str = "T4",
    training_gpu_hours: float = 0.0,
    training_gpu: str = "T4",
) -> LocalCost:
    """
    deployment: a key of GPU_HOURLY_USD, or "on-device" for zero marginal cost.
    """
    training_usd = training_gpu_hours * GPU_HOURLY_USD.get(training_gpu, 0.0)

    if deployment == "on-device":
        hourly = 0.0
        marginal_per_1k = 0.0
    else:
        hourly = GPU_HOURLY_USD.get(deployment)
        if hourly is None:
            raise ValueError(f"unknown deployment target {deployment!r}")
        marginal_per_1k = hourly * (seconds_per_page * 1000) / 3600.0

    total = training_usd + marginal_per_1k * pages / 1000.0
    return LocalCost(
        pages=pages,
        deployment=deployment,
        seconds_per_page=round(seconds_per_page, 3),
        gpu_hourly_usd=hourly,
        marginal_usd_per_1000_pages=round(marginal_per_1k, 4),
        training_usd=round(training_usd, 4),
        total_usd_at_volume=round(total, 4),
        usd_per_1000_pages_amortised=round(total / pages * 1000, 4) if pages else 0.0,
    )


def breakeven_pages(
    api_usd_per_page: float,
    local_marginal_usd_per_page: float,
    training_usd: float,
) -> float | None:
    """
    Page volume at which the local model's amortised cost crosses the API's.

    Returns None when the local marginal cost already exceeds the API's -- in
    which case there is no breakeven and the API wins at every volume. That is a
    real and reportable outcome, not a bug.
    """
    saving = api_usd_per_page - local_marginal_usd_per_page
    if saving <= 0:
        return None
    return training_usd / saving


def compare(
    pages: int,
    api_input_tokens: int,
    api_output_tokens: int,
    local_seconds_per_page: float,
    deployment: str = "T4",
    training_gpu_hours: float = 0.0,
    projection_volumes: tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000),
) -> dict:
    api = api_cost(pages, api_input_tokens, api_output_tokens)
    local = local_cost(pages, local_seconds_per_page, deployment, training_gpu_hours)

    api_pp = api.total_usd / pages if pages else 0.0
    local_pp = local.marginal_usd_per_1000_pages / 1000.0
    be = breakeven_pages(api_pp, local_pp, local.training_usd)

    projections = {}
    for v in projection_volumes:
        api_v = api_pp * v
        local_v = local.training_usd + local_pp * v
        projections[f"{v:,}_pages"] = {
            "api_usd": round(api_v, 2),
            "local_usd": round(local_v, 2),
            "ratio_api_over_local": round(api_v / local_v, 2) if local_v > 0 else None,
            "cheaper": "local" if local_v < api_v else "api",
        }

    return {
        "api": api.to_dict(),
        "local": local.to_dict(),
        "api_usd_per_page": round(api_pp, 6),
        "local_marginal_usd_per_page": round(local_pp, 6),
        "breakeven_pages": round(be) if be is not None else None,
        "breakeven_note": (
            "local marginal cost exceeds API per-page cost; API is cheaper at "
            "every volume on this deployment target"
            if be is None
            else "volume above which the local model is cheaper in total"
        ),
        "projections": projections,
        "caveats": [
            f"Gemini list price as of {GEMINI_PRICE_AS_OF}; batch API is 50% cheaper.",
            "GPU hourly rates are on-demand; committed/spot pricing is materially lower.",
            "Local marginal cost assumes the GPU is busy 100% of the time. Real "
            "utilisation on bursty traffic is often 10-30%, which multiplies the "
            "effective per-page cost by 3-10x.",
            "On-device deployment has zero marginal inference cost, but shifts cost "
            "to engineering, app size, and per-device performance variance.",
        ],
    }
