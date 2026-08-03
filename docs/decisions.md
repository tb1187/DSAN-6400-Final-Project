# Decisions log

Choices that are deliberate, non-obvious, or would look like bugs to someone
reading the code cold. Most of this is also methods-section material.

Corpus: `HOUSE_OVERSIGHT_009` — 2,897 documents, 23,124 page images, ~7.3M words.

---

## 1. Two chunk configurations, on purpose

| version | fingerprint | chunks | used for |
|---|---|---|---|
| v1 | `92dc44ec83930c20` | 28,583 | the extraction comparison **only** |
| v2 | `555a786298000384` | 29,671 (28,877 usable) | everything downstream |

v2 strips email headers from body text and prepends them to *every* body chunk of
their message, and never lets a body chunk cross a message boundary. It landed
after the extraction comparison had already been run and adjudicated, so the
comparison stays pinned to v1 rather than being re-run and re-annotated.

Reproduce the comparison with:

```bash
python scripts/run_extraction.py --system spacy \
    --chunks data/processed/chunks_v1_92dc44ec.parquet
```

Fingerprint any chunk table with:

```bash
python -c "
import pandas as pd, hashlib
c=pd.read_parquet('data/processed/chunks.parquet').sort_values('chunk_id')
print(hashlib.sha256('\n'.join(f'{r.chunk_id}:{r.char_start}:{r.char_end}' for r in c.itertuples()).encode()).hexdigest()[:16])"
```

Anything keyed to `chunk_id` — embeddings, mention tables — is only comparable to
things built on the same fingerprint.

### Chunk text ≠ chunk span (v2)

`chunk.text` is `prepended_header + body`, but `char_start`/`char_end` delimit only
the body. A position in the text maps into the document as
`char_start + (position - prefix_len)`. Mentions found *inside* a prepended header
get `doc_start = -1` and `note = "in_prepended_header"` — the header is a copy, so
the position is ambiguous, and that layer is covered deterministically anyway.
See `src/extraction/schema.py::chunk_offsets`.

---

## 2. Graph is built in three tiers

| tier | source | relations |
|---|---|---|
| 1 — deterministic | `email_headers.py` over raw text | `communicated with` |
| 2 — statistical | spaCy NER over chunks | PERSON / ORG / GPE nodes |
| 3 — rules | patterns over chunks | `represented by`, `employed by` |

Tier 1 gives 4,039 header blocks across 2,208 documents → 3,248 distinct messages,
4,088 sender→recipient edges, 984 actors, 97% with an exact page citation. The
load file itself tags only **64** emails; the rest are printed-to-PDF emails whose
headers survive in the OCR text. Do not build the communication network from
load-file metadata.

Weight edges by `nunique(message_key)`, not row count — 397 messages appear in more
than one document as forwarded copies, one of them 11 times.

### Tier 1 — resolving actors onto entities

984 email actors → 60% link to an NER entity (36 by address, 512 by exact name,
19 reordered, 16 via display name). The remaining 387 are **minted** as
`EML_*` nodes rather than dropped: NER failing to find a name is not evidence
that the correspondence did not happen, and dropping those edges would thin the
network silently. They hold 18% of message volume. `nodes.source` says which.

Names match on exact or reordered form only, never fuzzily — the value of this
tier is precision, and a wrong link fabricates a conversation. Single-token keys
(`weingarten`) *are* allowed on an exact alias hit, because resolution has
already decided with document-scoped evidence whether a bare surname belongs to
a full name.

Two artefacts worth knowing about:

* **Contested addresses.** Resolution attaches an address to every name that
  matched it, so `jeevacation@gmail.com` was claimed both by `jeffrey epstein`
  (24,678 mentions) and by `darren indyke jeffrey epstein` (2 mentions — a header
  string spaCy read as one person). The address goes to its dominant claimant.
* **FBI teletype routing blocks.** `FROM: MIAMI / TO: DIRECTOR / ATTN: SSA`
  parses like an email header but its actors are field offices. Rows carrying no
  address, subject *or* timestamp are dropped: 18 of 4,088, across 6 documents.
  Left in, `miami` ranked third by betweenness. `miami` and `1` still appear in
  the top ten — residual rows that do carry a subject. Flag them before reporting
  centrality.

### Tier 2 — co-occurrence weighted by NPMI

Two entities named in the same chunk. Raw co-occurrence counts rank by
*popularity* — every frequent entity looks related to every other — so edges are
weighted by normalised PMI, which asks whether a pair appears together more than
their individual rates predict.

| filter | reason |
|---|---|
| entity in ≥ 3 chunks, pair in ≥ 3 chunks | below this, association is unmeasurable |
| pair in ≥ **2 documents** | a pair confined to one document is a signature block repeating (dropped 25,143 pairs) |
| NPMI ≥ 0.2 | association no stronger than chance |
| chunks with > 60 entities dropped | guest lists and directories, where adjacency means nothing |
| entities > 6 tokens, or PERSON starting with a salutation | line-break spans (`arizona state university college of liberal arts and`) and `dear jeffrey`; flagged as `nodes.noisy`, not deleted |

Mentions inside a **prepended header are excluded** (28,548 of 540,586). v2
chunking copies a message's header onto every one of its body chunks, so counting
them multiplies one header's co-occurrences by the message's chunk count — and
tier 1 covers that layer deterministically anyway.

Result: 110,109 edges over 12,842 entities. **NPMI's top ranks are boilerplate** —
letterheads and bank disclosures score 1.0 because they only ever appear together.
That is the metric behaving correctly on repeated text, not a bug; rank by
`n_chunks` for reporting, and read NPMI as a filter rather than an importance
score.

---

## 3. Extraction comparison

**Sample.** 35 documents, stratified on `ocr_quality` × `text_style`, restricted to
150–1,500 words. The band is deliberate: proper-noun density is heavily skewed
(median 13/doc, mean 85, max 5,391), so unbanded sampling puts several thousand
mentions in the adjudication pool. Cost: court filings and book scans are
under-represented. `noisy/sparse` had only 3 documents in the frame and was dropped.

Per-stratum floors mean the sample is **not** corpus-representative — `clean/prose`
is 92% of the frame but 43% of the sample. Report the per-stratum table as the
primary result and the post-stratified estimate for corpus-level claims.

**Results** (corpus-weighted F1): llm **0.781**, hybrid **0.772**, spacy **0.719**.
The gap is concentrated in ORG precision (spacy 0.642, llm 0.785). Untyped scores
are much closer — spaCy finds the entities and mislabels them.

The hybrid is *candidate-guided*, not validate-only: the model may confirm, correct,
reject **and add**. Of its output, 674 confirmed / 274 corrected / **177 added**.
A validate-only hybrid could not exceed spaCy's recall by construction.

**Independent probe.** Email-header names give ground truth that owes nothing to the
pool: spacy 0.741, llm 0.845, hybrid 0.776 (n=58).

---

## 4. Adjudication rules

Judged per distinct `(document, surface, type)`, not per occurrence. Blinded and
shuffled; `pool_key.csv` holds the system attribution.

| # | case | rule |
|---|---|---|
| 1 | products/platforms (WhatsApp, Gmail) | not entities — not actors in the network |
| 2 | span boundaries | lenient — judge the entity, not the exact boundary |
| 3 | partial names (`Epstein`) | valid PERSON — resolution's job, not extraction's |
| 4 | titles (`President Trump`) | either span; `normalise_surface` strips titles |
| 5 | OCR damage | valid **if the entity is recoverable** (`Coutthouse`); not an entity if not (`reeegdt0 5t2 ie`) |
| 6 | header-contaminated spans (`McGyver\nSent`) | `n`, blank `true_type`, note `span` |
| 7 | street addresses | `n` — not GPE; letterhead, not actors |

Rule 5 was originally written without the recoverability clause, which is the
direct cause of the agreement problem below.

**Known limitation — inter-annotator agreement.** κ = 0.384 (n = 260 shared rows).
Disagreement concentrated in ORG (61% agreement vs 91% PERSON) and was
one-directional: 48 of 56 were one annotator accepting what the other rejected.
Not reconciled — a deliberate scoping decision for a course project.

Absolute figures are therefore approximate: per-system precision differed by 10–17
points between the two annotators' document sets. The **ranking** is stable —
identical under strict, lenient and exclusion resolution (F1 varying ≤ 0.017,
`--disagreements`) and within each annotator's documents. Report the ranking as the
finding; treat absolute values as indicative.

---

## 5. Production extraction uses spaCy, not the winner

The graph is built with spaCy corpus-wide: 540,586 mentions over 2,857 documents,
91,100 distinct `(surface, type)` pairs, 35 minutes, free. LLM extraction over the
same 28,877 chunks would cost ~$64 via the batch API.

This is a cost decision the comparison *licenses* rather than contradicts — it
quantifies the price (~6 F1 points, mostly ORG precision), which gets disclosed:

> Graph construction used spaCy extraction corpus-wide. Our evaluation indicates
> this underperforms LLM extraction by ~6 F1 points, so graph-derived results
> should be read as a lower bound.

Head/tail is extreme: 6,325 surface forms (7%) account for 68% of mentions, while
45,234 appear exactly once. Resolution effort belongs in the head.

---

## 6. Constraints for the RAG vs GraphRAG evaluation

1. **Uniform extraction tier across the corpus.** Spending LLM extraction only on
   documents that answer the demo questions would tune GraphRAG on its own test
   set — standard RAG doesn't use the graph, so better extraction helps one arm
   only. If a subset must be targeted, target by an intrinsic property decided in
   advance, never by which documents the questions touch.
2. **Fix the question set first**, derived from `email_edges.parquet` where the
   answer path is verifiable. Writing questions after seeing what the graph can do
   is the same contamination through a different door.
3. **Share everything except retrieval.** Same model, prompt, `max_tokens` and
   quality filters in both arms; only `retrieve(question, k) -> [chunk_id]` differs.

---

## 7. What is tracked in git

Committed — expensive or impossible to recreate, and small (~1.2 MB):
`results/adjudication/**.csv` (≈1,568 human judgements + pool key),
`mentions_{llm,hybrid,spacy}.parquet` (paid API calls), `eval_sample.parquet`
(defines which documents the labels refer to).

Ignored — deterministically regenerable in ~40 minutes, and large (68 MB):
`chunks*.parquet`, `mentions_spacy_corpus.parquet`, `manifest.parquet`,
`text_stats.parquet`. Share these via the team Drive with the chunk fingerprint in
the filename.

Rebuild order:

```bash
python scripts/build_manifest.py       # ~50s
python scripts/build_chunks.py         # ~35s
python scripts/build_email_edges.py    # ~30s
python scripts/run_extraction.py --system spacy --scope corpus   # ~35 min
```
