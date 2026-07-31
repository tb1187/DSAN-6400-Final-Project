"""Cheap, dependency-light text profiling used by the EDA.

The production mixes clean digital text (emails, exported PDFs) with badly OCR'd
scans of books and faxes. Everything downstream — chunking, NER, retrieval —
depends on knowing which is which, so we score every document up front.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]+")
LINE_RE = re.compile(r"\r?\n")

# Bates stamps burned into the page images resurface in the OCR text.
BATES_STAMP_RE = re.compile(r"HOUSE[\s_]*OVERSIGHT[\s_]*\d{4,}", re.IGNORECASE)

# A small, high-frequency English vocabulary: the share of tokens falling in it
# is a good proxy for OCR quality without pulling in a spell-checker.
COMMON_WORDS = set(
    """the of and to in a is that it for was on as with he be at by i this had not
    are but from or have an they which one you were her all she there would their we
    him been has when who will no more if out so said what up its about into than them
    can only other new some could time these two may then do first any my now such like
    our over man me even most made after also did many before must through back years
    where much your way well down should because each just those people mr how too little
    state good very make world still own see men work long get here between both life
    being under never day same another know while last might us great old year off
    come since against go came right used take three""".split()
)

REDACTION_RE = re.compile(
    r"_{4,}|\[redacted\]|\bredacted\b|\bnon-responsive\b|\bprivileged\s*-\s*redacted\b",
    re.IGNORECASE,
)

# Legitimately vowel-less tokens — file extensions, protocols and abbreviations that
# turn up constantly in email headers and would otherwise read as OCR garble.
VOWELLESS_WORDS = {
    "pdf", "png", "ppt", "pptx", "xls", "xlsx", "txt", "csv", "htm", "html", "https",
    "www", "fwd", "bcc", "llp", "llc", "nyc", "fyi", "pls", "thx", "tks", "mrs", "hrs",
    "gmt", "est", "pst", "cst", "edt", "pdt", "cdt", "mdt", "jpg", "gif", "zip", "wsj",
    "nyt", "cnn", "cbs", "nbc", "bbc", "gtc", "phd", "dept", "mgmt", "pymt", "rcvd",
}

EMAIL_RE = re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.\-]+")
PHONE_RE = re.compile(r"\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
)
# Rough proper-noun candidate: one or more capitalised tokens in sequence.
PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-z]{1,}\.?\s?){2,}")


def strip_bates(text: str) -> str:
    """Remove burned-in Bates stamps from OCR text."""
    return BATES_STAMP_RE.sub(" ", text)


# Characters that are legitimately part of a document: letters, digits, whitespace,
# any punctuation or dash, currency symbols and formatting marks. Anything else is
# treated as scanning noise.
_OK_UNICODE_CATEGORIES = {
    "Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Sc", "Sm", "Cf",
}


def _is_expected_char(c: str) -> bool:
    return c.isalnum() or c.isspace() or unicodedata.category(c) in _OK_UNICODE_CATEGORIES


def _is_vowelless(word: str) -> bool:
    return (
        len(word) >= 3
        and not set(word.lower()) & set("aeiouy")
        and word.lower() not in VOWELLESS_WORDS
    )


def profile_text(text: str) -> dict:
    """Return a dict of per-document text statistics."""
    body = strip_bates(text)
    # URLs and email addresses are tokenised separately: left in place they look
    # like garble to the word-level metrics ("politico.com/story/2o17/o2/...").
    prose = EMAIL_RE.sub(" ", URL_RE.sub(" ", body))
    words = WORD_RE.findall(prose)
    lower = [w.lower() for w in words]
    n_words = len(words)
    lines = [ln for ln in LINE_RE.split(body)]
    nonblank = [ln for ln in lines if ln.strip()]
    alpha = sum(c.isalpha() for c in body)
    printable = sum((not c.isspace()) for c in body)

    counts = Counter(lower)
    common_hits = sum(counts[w] for w in COMMON_WORDS if w in counts)

    return {
        "n_chars": len(text),
        "n_words": n_words,
        "n_lines": len(nonblank),
        "n_unique_words": len(counts),
        "type_token_ratio": len(counts) / n_words if n_words else 0.0,
        "mean_word_len": sum(len(w) for w in words) / n_words if n_words else 0.0,
        "mean_line_len": (
            sum(len(ln) for ln in nonblank) / len(nonblank) if nonblank else 0.0
        ),
        "alpha_ratio": alpha / printable if printable else 0.0,
        "upper_ratio": (
            sum(c.isupper() for c in body if c.isalpha()) / alpha if alpha else 0.0
        ),
        # OCR quality proxies
        "common_word_ratio": common_hits / n_words if n_words else 0.0,
        "short_token_ratio": (
            sum(len(w) <= 2 for w in words) / n_words if n_words else 0.0
        ),
        # Tokens of 3+ letters with no vowel ("grnail", "Xa-n.adu") are almost
        # always OCR garble, and unlike common_word_ratio this holds up on very
        # short documents such as header-only emails.
        "no_vowel_ratio": (
            sum(_is_vowelless(w) for w in words) / n_words if n_words else 0.0
        ),
        # The same measure over distinct types rather than tokens. A document that
        # repeats one abbreviation ("Fmr", 89 times in a guest list) is not garbled;
        # genuine OCR damage produces many *different* nonsense tokens.
        "no_vowel_type_ratio": (
            sum(_is_vowelless(w) for w in counts) / len(counts) if counts else 0.0
        ),
        # Runs of underscores are how this production renders redactions, so they
        # are counted separately rather than as garbled characters.
        "nonword_char_ratio": (
            sum(not _is_expected_char(c) for c in body) / printable if printable else 0.0
        ),
        "n_redaction_marks": len(REDACTION_RE.findall(body)),
        # Content signals
        "n_emails": len(EMAIL_RE.findall(body)),
        "n_phones": len(PHONE_RE.findall(body)),
        "n_urls": len(URL_RE.findall(body)),
        "n_dates": len(DATE_RE.findall(body)),
        "n_proper_noun_spans": len(PROPER_NOUN_RE.findall(body)),
        "n_bates_stamps": len(BATES_STAMP_RE.findall(text)),
    }


def profile_corpus(paths: pd.Series, loader) -> pd.DataFrame:
    """Profile many documents; ``loader`` maps a path to its text."""
    records = []
    for doc_id, path in paths.items():
        if not isinstance(path, str):
            records.append({"doc_id": doc_id})
            continue
        stats = profile_text(loader(path))
        stats["doc_id"] = doc_id
        records.append(stats)
    return pd.DataFrame(records).set_index("doc_id")


def ocr_quality_flag(row) -> str:
    """Grade *character-level* OCR integrity: ``empty`` / ``clean`` / ``noisy`` / ``poor``.

    Deliberately built only on garble signals (vowel-less tokens, out-of-set
    characters), because they mean the same thing on a 30-word email and on a
    30-page transcript. Whether a document reads as running prose is a separate
    question — see :func:`text_style_flag`.
    """
    n_words = row.get("n_words") or 0
    if n_words < 5:
        return "empty"
    # Cuts sit near the 99th / 95th percentile of no_vowel_type_ratio and the
    # 99.5th / 99th of nonword_char_ratio across this production. Short documents
    # get a looser vowel cut: in a 30-word header block, two acronyms would
    # otherwise clear the threshold on their own.
    poor_cut, noisy_cut = (0.12, 0.06) if n_words < 100 else (0.06, 0.03)
    if row["no_vowel_type_ratio"] > poor_cut or row["nonword_char_ratio"] > 0.005:
        return "poor"
    if row["no_vowel_type_ratio"] > noisy_cut or row["nonword_char_ratio"] > 0.001:
        return "noisy"
    return "clean"


def text_style_flag(row) -> str:
    """Describe *composition*: ``short`` / ``prose`` / ``mixed`` / ``sparse``.

    ``common_word_ratio`` needs a few hundred words before it stabilises, so short
    documents get their own bucket rather than a bad grade. Below 150 words the
    ratio is not read at all.

    A long document with few function words is not necessarily badly scanned — it
    is usually a signature block, a financial table or a recipient list. It *can*
    also be a page scan whose words were truncated at the line edge, which the
    character-level metrics miss entirely, so ``sparse`` is a review flag rather
    than a verdict. Cuts sit near the 2nd and 10th percentiles of
    ``common_word_ratio`` among 150+ word documents in this production.
    """
    n_words = row.get("n_words") or 0
    if n_words < 150:
        return "short"
    if row["common_word_ratio"] < 0.20:
        return "sparse"
    if row["common_word_ratio"] < 0.33:
        return "mixed"
    return "prose"


def usable_for_extraction(row) -> bool:
    """Whether a document should be fed to entity/relationship extraction."""
    return row.get("ocr_quality") not in {"poor", "empty"}


DOC_TYPE_PATTERNS = {
    "court_filing": re.compile(
        r"\b(IN THE (?:CIRCUIT|DISTRICT|SUPREME) COURT|UNITED STATES DISTRICT COURT|"
        r"CASE NO|Plaintiff|Defendant|MOTION TO|MEMORANDUM (?:OF|IN)|AFFIDAVIT|SUBPOENA)\b",
        re.IGNORECASE,
    ),
    "deposition": re.compile(
        r"\b(DEPOSITION OF|EXAMINATION BY|Q\.\s|A\.\s|VIDEOTAPED DEPOSITION|"
        r"CERTIFIED (?:COURT )?REPORTER)\b"
    ),
    "email": re.compile(
        r"(^|\n)\s*(From:|To:|Sent:|Subject:|Cc:)\s", re.IGNORECASE | re.MULTILINE
    ),
    "correspondence": re.compile(
        r"\b(Dear (?:Mr|Ms|Mrs|Dr|Judge)\.?|Sincerely,|Very truly yours|VIA (?:FACSIMILE|"
        r"FEDERAL EXPRESS|EMAIL))\b",
        re.IGNORECASE,
    ),
    "financial": re.compile(
        r"\b(Invoice|Wire Transfer|Account (?:No|Number)|Balance Sheet|USD [\d,]+|"
        r"Statement of Account|Trust Agreement)\b",
        re.IGNORECASE,
    ),
    "flight_or_travel": re.compile(
        r"\b(Flight (?:No|Log|Manifest)|Passenger|Departure|Arrival|Tail Number|"
        r"N\d{3}[A-Z]{2}\b|Itinerary)\b",
        re.IGNORECASE,
    ),
    "news_or_book": re.compile(
        r"\b(Chapter \w+|Copyright ©|All rights reserved|Vanity Fair|The New York Times|"
        r"Associated Press)\b"
    ),
}


def classify_doc_type(text: str) -> list[str]:
    """Weak, rule-based document typing. Returns every pattern family that fires."""
    return [name for name, rx in DOC_TYPE_PATTERNS.items() if rx.search(text)]


__all__ = [
    "profile_text",
    "profile_corpus",
    "ocr_quality_flag",
    "text_style_flag",
    "usable_for_extraction",
    "classify_doc_type",
    "strip_bates",
    "DOC_TYPE_PATTERNS",
]
