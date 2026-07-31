"""
Detect source files that are actually the same underlying document.

Why this exists: the 15 supplied samples are not 15 independent documents.
`train_sample_9` and `train_sample_10` are contiguous slices of one 90-page bill
(Bill No INT2043376). Splitting related files across the train/test boundary
leaks the same document, template, patient and scanner into both sides, which
inflates held-out accuracy and invalidates the benchmark.

This finds candidates two ways:

  1. Shared identifiers in the PDF text layer (bill numbers, IP/UHID numbers).
     Precise, but only works on the 3 of 15 files that have a real text layer.
  2. "Page N of M" markers. Two files reporting the same M and adjacent N
     ranges are almost certainly one document.

The 12 scanned files cannot be checked this way. Either OCR them first or
eyeball the letterheads -- and until you have, treat the held-out numbers as
provisional. Add confirmed pairs to config.DOCUMENT_GROUPS.

Usage:
    python -m vlm.data.find_duplicates --pdf-dir /tmp/pdfs
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import DOCUMENT_GROUPS, REPO_ROOT  # noqa: E402

ID_PATTERNS = [
    re.compile(r"bill\s*no\.?\s*:?\s*([A-Z0-9][A-Z0-9./-]{5,})", re.I),
    re.compile(r"ip\s*(?:no|number)\.?\s*:?\s*([A-Z0-9][A-Z0-9./-]{5,})", re.I),
    re.compile(r"uhid\s*:?\s*([A-Z0-9][A-Z0-9./-]{5,})", re.I),
    re.compile(r"reg(?:n)?\.?\s*no\.?\s*:?\s*([A-Z0-9][A-Z0-9./-]{5,})", re.I),
]
PAGE_OF = re.compile(r"page\s*(\d+)\s*(?:of|/)\s*(\d+)", re.I)


def pdf_text(path: Path) -> str:
    try:
        return subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, default=None)
    ap.add_argument("--zip", type=Path, default=REPO_ROOT / "attachments.zip")
    args = ap.parse_args()

    with TemporaryDirectory() as tmp:
        if args.pdf_dir:
            pdf_dir = args.pdf_dir
        else:
            pdf_dir = Path(tmp)
            with zipfile.ZipFile(args.zip) as zf:
                zf.extractall(pdf_dir)

        pdfs = sorted(Path(pdf_dir).rglob("*.pdf"), key=lambda p: p.name)
        ids: dict[str, set[str]] = {}
        totals: dict[str, list[tuple[int, int]]] = {}
        no_text = []

        for p in pdfs:
            text = pdf_text(p)
            if len(text.strip()) < 100:
                no_text.append(p.name)
            found = set()
            for pat in ID_PATTERNS:
                found |= {m.strip(" .:-") for m in pat.findall(text)}
            ids[p.name] = found
            totals[p.name] = [(int(a), int(b)) for a, b in PAGE_OF.findall(text)]

        print(f"{len(pdfs)} files; {len(no_text)} have no usable text layer "
              f"and CANNOT be auto-checked:\n  {', '.join(no_text)}\n")

        by_id = defaultdict(set)
        for name, found in ids.items():
            for i in found:
                by_id[i].add(name)

        pairs = defaultdict(set)
        for ident, names in by_id.items():
            if len(names) > 1:
                for n in names:
                    pairs[frozenset(names)].add(ident)

        if pairs:
            print("SHARED IDENTIFIERS -- almost certainly the same document:")
            for names, idents in pairs.items():
                joined = ", ".join(sorted(names))
                print(f"  {joined}")
                print(f"    via: {', '.join(sorted(idents))}")
                for n in sorted(names):
                    rng = totals.get(n) or []
                    if rng:
                        print(f"    {n}: pages {min(a for a, _ in rng)}-"
                              f"{max(a for a, _ in rng)} of {rng[0][1]}")
        else:
            print("No shared identifiers found in the text-bearing files.")

        print("\nCurrently declared in config.DOCUMENT_GROUPS:")
        if DOCUMENT_GROUPS:
            for f, g in sorted(DOCUMENT_GROUPS.items()):
                print(f"  {f} -> {g}")
        else:
            print("  (none)")

        undeclared = [
            sorted(names) for names in pairs
            if len({DOCUMENT_GROUPS.get(n, n) for n in names}) > 1
        ]
        if undeclared:
            print("\n  ACTION REQUIRED: these detected groups are not declared:")
            for g in undeclared:
                print(f"    {g}")
            return 1

        print("\nAll auto-detected groups are declared. Scanned files still need "
              "a human pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
