"""
List the Gemini models this API key can actually call.

Run this whenever extraction fails with a 404. A 404 on generateContent means
the model name does not exist for your key and API version -- it does NOT mean
the key is invalid, which is why the request still authenticates and still fails.

    python list_models.py

Copy a working name into .env as GEMINI_MODEL, and two more as
GEMINI_MODEL_FALLBACKS.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
if not key:
    sys.exit("GOOGLE_API_KEY not set (checked .env and the environment)")

try:
    import google.generativeai as genai
except ImportError:
    sys.exit("google-generativeai is not installed: pip install google-generativeai")

genai.configure(api_key=key)

print(f"key ...{key[-4:]}  |  SDK google-generativeai\n")

usable = []
try:
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", [])
        if "generateContent" in methods:
            usable.append(m.name)
except Exception as e:  # noqa: BLE001
    sys.exit(f"Could not list models: {type(e).__name__}: {e}")

if not usable:
    sys.exit("No models on this key support generateContent.")

print(f"{len(usable)} model(s) support generateContent:\n")
for name in usable:
    print(f"  {name}")

# The extractor passes bare names, not the "models/" prefix.
short = [n.split("/", 1)[-1] for n in usable]

# Exclude specialised variants: image *generators*, TTS, robotics, research
# agents and computer-use models all advertise generateContent but are not
# general vision-language models and will not extract a bill table.
EXCLUDE = ("tts", "image", "robotics", "computer-use", "deep-research",
           "lyria", "nano-banana", "omni", "antigravity", "customtools")
general = [n for n in short if not any(x in n for x in EXCLUDE)]

# Prefer flash tier: this is a high-volume page-by-page workload, and pro
# costs several times more for a task that is mostly transcription.
flash = [n for n in general if "flash" in n and "lite" not in n]
pro = [n for n in general if "pro" in n]


def _rank(name: str) -> tuple:
    """
    Sort key, best first.

    Ordering by the API's own list order is a trap: it returns the oldest
    families first, so the naive "take the first flash model" picks a
    generation that may already be closed to new users. Rank by version
    descending, and prefer stable over preview.
    """
    import re

    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    version = float(m.group(1)) if m else 0.0
    is_latest_alias = name.endswith("-latest")
    is_preview = "preview" in name
    # Aliases track the current model, so they never go stale -- but an
    # explicit version is more reproducible. Rank aliases just below the
    # newest explicit stable release.
    return (version, not is_preview, is_latest_alias)


flash_ranked = sorted(flash, key=_rank, reverse=True)
pro_ranked = sorted(pro, key=_rank, reverse=True)

print("\nSuggested .env entries:")
if flash_ranked:
    print(f"  GEMINI_MODEL={flash_ranked[0]}")
    others = flash_ranked[1:3]
    if others:
        print(f"  GEMINI_MODEL_FALLBACKS={','.join(others)}")
elif pro_ranked:
    print(f"  GEMINI_MODEL={pro_ranked[0]}")

print(
    "\nNotes:"
    "\n  - This project sends page IMAGES, so pick a multimodal model. Text-only"
    "\n    models authenticate fine and then fail on the image parts."
    "\n  - Skip *-image models: those GENERATE images, they do not read them."
    "\n  - Flash tier is the right default; pro costs several times more for"
    "\n    what is mostly transcription."
)

# ── Live probe ─────────────────────────────────────────────────────────────
# Listing models and generating content are DIFFERENT permission paths. A
# project that has been denied access will happily return the full model list
# and then 403 on every generateContent call -- which is exactly how this
# project's failure presented. So actually make a call. It costs a few tokens.
probe_targets = (flash_ranked + pro_ranked)[:6]
if not probe_targets:
    sys.exit(0)

print("\n" + "=" * 60)
print("Probing generateContent (the call that actually matters)\n")

working, retired, throttled, denied = [], [], [], []
for name in probe_targets:
    try:
        resp = genai.GenerativeModel(name).generate_content("Reply with the single word: ok")
        text = (resp.text or "").strip().replace("\n", " ")[:30]
        print(f"  {name:<30} OK       -> {text!r}")
        working.append(name)
        if len(working) >= 3:
            break
    except Exception as e:  # noqa: BLE001
        kind = type(e).__name__
        msg = str(e)
        low = msg.lower()
        if "no longer available" in low or "not found" in low or "404" in low:
            label, bucket = "RETIRED ", retired
        elif "429" in low or "quota" in low or "exhausted" in low:
            label, bucket = "NO QUOTA", throttled
        elif "403" in low or "denied" in low or "permission" in low:
            label, bucket = "DENIED  ", denied
        else:
            label, bucket = "FAILED  ", []
        bucket.append(name)
        print(f"  {name:<30} {label} {kind}: {msg[:110]}")

print()
if working:
    print(f"WORKING: {', '.join(working)}\n")
    print("Put this in .env:")
    print(f"  GEMINI_MODEL={working[0]}")
    if len(working) > 1:
        print(f"  GEMINI_MODEL_FALLBACKS={','.join(working[1:3])}")
    print("  USE_MOCK_MODE=false")
    print("\nThen fully restart uvicorn (a --reload does not re-read .env).")
else:
    print("No model on this key could generate content.")
    if retired:
        print(f"  retired / unavailable to new users: {', '.join(retired)}")
    if throttled:
        print(f"  no quota on free tier:              {', '.join(throttled)}")
    if denied:
        print(f"  project denied access:              {', '.join(denied)}")
    if denied and not (retired or throttled):
        print("\n  -> Project-level block. Another key from the same project will")
        print("     fail identically; you need a different Cloud project.")
    elif throttled:
        print("\n  -> Free tier gives 0 quota on these. Enable billing, or try a")
        print("     lite/flash model with a free allocation.")
    sys.exit(1)
