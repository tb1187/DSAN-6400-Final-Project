"""Readers for the Concordance production files shipped with the House Oversight release.

The production follows the standard eDiscovery layout:

* ``DATA/*.dat``  – Concordance delimited load file, one row per document,
  UTF-8 with BOM, field delimiter ``\\x14`` (ASCII 20), text qualifier ``þ`` (ASCII 254).
* ``DATA/*.opt``  – Opticon image cross-reference, one row per *page* image.
  Column 4 is ``Y`` on the first page of each document, column 7 carries the
  document page count on that same row.
* ``TEXT/<vol>/<bates>.txt`` – extracted / OCR'd text, one file per document,
  named after the document's first Bates number.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DAT_DELIMITER = "\x14"
DAT_QUOTECHAR = "þ"

# Columns in the .dat that hold dates in mm/dd/YYYY form.
DATE_COLUMNS = [
    "Date Created",
    "Date Last Modified",
    "Date Received",
    "Date Sent",
]

NUMERIC_COLUMNS = ["Pages", "File Size"]


@dataclass(frozen=True)
class Production:
    """Paths to one extracted production volume."""

    root: Path

    @property
    def dat_path(self) -> Path:
        return next((self.root / "DATA").glob("*.dat"))

    @property
    def opt_path(self) -> Path:
        return next((self.root / "DATA").glob("*.opt"))

    @property
    def text_root(self) -> Path:
        return self.root / "TEXT"

    @property
    def natives_root(self) -> Path:
        return self.root / "NATIVES"


def read_dat(path: str | Path) -> pd.DataFrame:
    """Read a Concordance ``.dat`` load file into a DataFrame of strings."""
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = list(
        csv.reader(io.StringIO(text), delimiter=DAT_DELIMITER, quotechar=DAT_QUOTECHAR)
    )
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df.apply(lambda s: s.str.strip())


def read_opt(path: str | Path) -> pd.DataFrame:
    """Read an Opticon ``.opt`` file (one row per page image)."""
    cols = [
        "bates",
        "volume",
        "image_path",
        "doc_break",
        "box_break",
        "folder_break",
        "page_count",
    ]
    df = pd.read_csv(path, header=None, names=cols, dtype=str).fillna("")
    df["page_count"] = pd.to_numeric(df["page_count"], errors="coerce")
    return df


def page_counts_from_opt(opt: pd.DataFrame) -> pd.DataFrame:
    """Per-document page counts, derived two ways.

    ``opt_pages`` is the count declared on the document-break row; ``image_pages``
    counts the image rows actually present. They should agree.
    """
    starts = opt[opt["doc_break"].str.upper() == "Y"]
    opt = opt.copy()
    opt["doc_id"] = opt["bates"].where(opt["doc_break"].str.upper() == "Y").ffill()
    image_pages = opt.groupby("doc_id").size().rename("image_pages")
    declared = (
        starts.set_index("bates")["page_count"].rename("opt_pages").rename_axis("doc_id")
    )
    return pd.concat([declared, image_pages], axis=1).reset_index()


def _text_index(text_root: str | Path) -> pd.DataFrame:
    """Map each extracted text file to its Bates number and volume."""
    files = sorted(Path(text_root).rglob("*.txt"))
    return pd.DataFrame(
        {
            "doc_id": [f.stem for f in files],
            "text_path": [str(f) for f in files],
            "volume": [f.parent.name for f in files],
            "text_bytes": [f.stat().st_size for f in files],
        }
    )


def load_text(path: str | Path) -> str:
    """Read one extracted text file, tolerating the BOM and stray encodings."""
    return Path(path).read_text(encoding="utf-8-sig", errors="replace")


def build_manifest(root: str | Path) -> pd.DataFrame:
    """Join load-file metadata, Opticon page counts and extracted text paths.

    One row per document, keyed on ``doc_id`` (the beginning Bates number).
    """
    prod = Production(Path(root))
    dat = read_dat(prod.dat_path)
    opt = read_opt(prod.opt_path)

    df = dat.rename(
        columns={
            "Bates Begin": "doc_id",
            "Bates End": "bates_end",
            "Bates Begin Attach": "attach_begin",
            "Bates End Attach": "attach_end",
            "Custodian/Source": "custodian",
            "Document Extension": "extension",
            "Original Filename": "filename",
            "Email Subject/Title": "email_subject",
            "Email From": "email_from",
            "Email To": "email_to",
            "Email CC": "email_cc",
            "Email BCC": "email_bcc",
            "Parent Document ID": "parent_doc_id",
            "Document Title": "doc_title",
            "MD5 Hash": "md5",
            "File Size": "file_size",
            "Text Link": "text_link",
            "Native Link": "native_link",
            "Attachment Document": "attachment_document",
        }
    )
    df.columns = [c.lower().replace(" ", "_").replace("/", "_") for c in df.columns]
    df = df.replace("", pd.NA)

    for col in NUMERIC_COLUMNS:
        key = col.lower().replace(" ", "_")
        if key in df:
            df[key] = pd.to_numeric(df[key], errors="coerce")
    for col in DATE_COLUMNS:
        key = col.lower().replace(" ", "_")
        if key in df:
            df[key] = pd.to_datetime(df[key], format="%m/%d/%Y", errors="coerce")

    df = df.merge(page_counts_from_opt(opt), on="doc_id", how="left")
    df = df.merge(_text_index(prod.text_root), on="doc_id", how="left")

    df["bates_seq"] = pd.to_numeric(
        df["doc_id"].str.rsplit("_", n=1).str[-1], errors="coerce"
    )
    df["has_text"] = df["text_path"].notna()
    df["has_native"] = df["native_link"].notna()
    df["is_email"] = df["email_from"].notna()
    df["is_attachment"] = df["parent_doc_id"].notna()

    return df.sort_values("bates_seq").reset_index(drop=True)


__all__ = [
    "Production",
    "read_dat",
    "read_opt",
    "page_counts_from_opt",
    "build_manifest",
    "load_text",
]
