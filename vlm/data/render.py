"""
Render the source PDFs to per-page PNGs.

Kept deliberately separate from labelling so that rendering is deterministic and
cached: every downstream stage addresses a page by `{stem}_p{n:03d}.png` and no
stage re-rasterises.

Usage:
    python -m vlm.data.render --zip attachments.zip
    python -m vlm.data.render --pdf-dir some/dir --dpi 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import PAGES_DIR, REPO_ROOT  # noqa: E402


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def render_pdf(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    """Rasterise one PDF to PNGs. Prefers poppler, falls back to PyMuPDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    if _have("pdftoppm"):
        prefix = out_dir / stem
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True,
            capture_output=True,
        )
        # poppler emits {stem}-1.png / {stem}-01.png depending on page count
        produced = sorted(out_dir.glob(f"{stem}-*.png"))
        renamed = []
        for p in produced:
            page_no = int(p.stem.rsplit("-", 1)[1])
            target = out_dir / f"{stem}_p{page_no:03d}.png"
            p.rename(target)
            renamed.append(target)
        return renamed

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "Neither poppler (pdftoppm) nor PyMuPDF is available. "
            "Install one: `apt-get install poppler-utils` or `pip install pymupdf`."
        )

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    out = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        target = out_dir / f"{stem}_p{i:03d}.png"
        pix.save(target)
        out.append(target)
    doc.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=REPO_ROOT / "attachments.zip")
    ap.add_argument("--pdf-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=PAGES_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.pdf_dir:
            pdf_dir = args.pdf_dir
        else:
            if not args.zip.exists():
                print(f"error: {args.zip} not found", file=sys.stderr)
                return 1
            pdf_dir = Path(tmp)
            with zipfile.ZipFile(args.zip) as zf:
                zf.extractall(pdf_dir)

        pdfs = sorted(Path(pdf_dir).rglob("*.pdf"))
        if not pdfs:
            print("error: no PDFs found", file=sys.stderr)
            return 1

        manifest = []
        for pdf in pdfs:
            pages = render_pdf(pdf, args.out, dpi=args.dpi)
            for p in pages:
                manifest.append(
                    {
                        "page_id": p.stem,
                        "source_pdf": pdf.name,
                        "page_no": int(p.stem.rsplit("_p", 1)[1]),
                        "image_path": str(p.relative_to(args.out.parent.parent)
                                          if args.out.is_relative_to(args.out.parent.parent)
                                          else p),
                        "abs_path": str(p.resolve()),
                    }
                )
            print(f"{pdf.name}: {len(pages)} pages")

        manifest_path = args.out / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n{len(manifest)} pages -> {args.out}")
        print(f"manifest -> {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
