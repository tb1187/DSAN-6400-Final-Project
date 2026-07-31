"""Build the document manifest from the extracted production.

Usage:
    python scripts/build_manifest.py [--root data/raw/extracted] [--out data/processed]

Writes ``manifest.parquet`` (metadata + page counts + text paths) and
``text_stats.parquet`` (per-document text profiling) to the output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.loadfile import build_manifest, load_text  # noqa: E402
from src.ingestion.textstats import (  # noqa: E402
    classify_doc_type,
    ocr_quality_flag,
    profile_corpus,
    text_style_flag,
    usable_for_extraction,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=REPO_ROOT / "data" / "raw" / "extracted")
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.root)
    print(f"manifest: {len(manifest):,} documents, {manifest.shape[1]} columns")

    stats = profile_corpus(
        manifest.set_index("doc_id")["text_path"], loader=load_text
    )
    stats["ocr_quality"] = stats.apply(ocr_quality_flag, axis=1)
    stats["text_style"] = stats.apply(text_style_flag, axis=1)
    stats["usable"] = stats.apply(usable_for_extraction, axis=1)
    types = {
        doc_id: classify_doc_type(load_text(path))
        for doc_id, path in manifest.set_index("doc_id")["text_path"].items()
        if isinstance(path, str)
    }
    stats["doc_types"] = stats.index.map(lambda d: "|".join(types.get(d, [])))

    manifest.to_parquet(out / "manifest.parquet", index=False)
    stats.reset_index().to_parquet(out / "text_stats.parquet", index=False)
    print(f"wrote {out / 'manifest.parquet'} and {out / 'text_stats.parquet'}")
    print(stats["ocr_quality"].value_counts().to_string())
    print(stats["text_style"].value_counts().to_string())


if __name__ == "__main__":
    main()
