# Network Analysis and Graph RAG of the Epstein Files

DSAN 6400 Final Project. Tyler Blue, Andrew Moy, Ethan Wotring.

We take 2,897 documents from the House Oversight Committee's November 2025 release,
reconstruct the communication network buried in their OCR text, and use the
resulting knowledge graph to test whether graph-augmented retrieval beats ordinary
vector search.

The corpus arrives as scanned pages with eDiscovery load files. The load files tag
only **64** documents as email, while the OCR text contains **4,039** email header
blocks across 2,208 documents. Everything downstream follows from that gap.

## Results

| | |
|---|---|
| Knowledge graph | 81,796 nodes, 113,613 edges, 4 relation types |
| Communication network | 850 actors, 1,132 directed edges, dated 2006 to 2019 |
| Entity extraction (F1) | LLM 0.78, hybrid 0.77, spaCy 0.72 |
| GraphRAG vs RAG, one hop | 6.8 vs 3.6 |
| GraphRAG vs RAG, single entity | 6.8 vs 7.4 |

The full write-up is [`docs/6400-Final-Report.qmd`](docs/6400-Final-Report.qmd).
Design decisions and their justifications are in
[`docs/decisions.md`](docs/decisions.md).

## Structure

```
src/
  ingestion/         load files, OCR profiling, normalisation, chunking, email headers
  extraction/        entity extraction (spaCy / LLM / hybrid) and shared schema
  knowledge_graph/   entity resolution, edge construction, traversal, graph store
  retrieval/         embeddings, FAISS index, RAG and GraphRAG pipelines
  evaluation/        retrieval metrics, faithfulness checks, gold question set
scripts/             one runnable step each, all take --help
notebooks/           EDA and network analysis
tests/               pytest, 55 tests
docs/                report, decisions log, bibliography
results/             adjudication labels, figures, scores
```

Key modules in `src/retrieval/`:

- `embeddings.py` embeds passages or a query with BAAI/bge-small-en-v1.5.
- `index.py` builds, saves and loads the FAISS index.
- `retriever.py` embeds a query and returns the top k passages.
- `rag.py` runs the full pipeline, from vector store loading through the LLM call.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Set `ANTHROPIC_API_KEY` in your environment or in a `.env` file (see `.env.example`).
Only the LLM extraction arms and the RAG answer generation need it.

## Usage

Rebuilding everything from the raw release takes about 40 minutes.

```bash
python scripts/build_manifest.py                                  # ~50s
python scripts/build_chunks.py                                    # ~35s
python scripts/build_email_edges.py                               # ~30s
python scripts/run_extraction.py --system spacy --scope corpus    # ~35 min
python scripts/resolve_entities.py                                # ~1 min
python scripts/build_edges.py                                     # ~3 min
```

`resolve_entities.py` and `build_edges.py` must run together. Entity ids come from
sort order, so an edge table built against a stale entity table points at ids that
no longer mean the same thing.

For retrieval:

```bash
python scripts/build_index.py     # embeds chunks, writes the FAISS index
python scripts/ask.py             # interactive query loop
```

`nodes.parquet` and `edges.parquet` are committed, so the network analysis runs
without rebuilding anything. The larger derived files are not in git. They
regenerate from the commands above, or ask for the Drive link.

## Use of AI

We used AI throughout this project, in three distinct ways. We are setting them out
separately because they carry different weight.

**As an object of study.** `claude-sonnet-5` is one of the three entity extraction
systems we evaluated, and it generates answers in both the RAG and GraphRAG arms.
An LLM also acted as judge in the retrieval comparison, alongside human review.
These uses are the subject of the experiment and are documented in the report.

**To expand what we could attempt.** Several parts of this project were beyond what
we would have scoped without AI assistance. The offset-preserving text normaliser,
which lets any extracted entity map back to an exact page in the original scan, is
the clearest example. So is the three-way extraction comparison with pooled blind
adjudication, which we would likely have replaced with an unevaluated choice of a
single extractor. The graph would have been smaller, less well justified, and
harder to cite.

**To execute the work.** We used Claude Code as a development collaborator across
most of the codebase. It wrote and reviewed code in the ingestion, extraction,
resolution and edge construction modules, proposed the sampling and adjudication
design for the extraction evaluation, and helped draft and format this report. It
also caught defects that would otherwise have reached our results, including an
email address attached to the wrong person during entity resolution and a header
parsing bug that split one journalist into two apparently corresponding nodes.

**What we did ourselves.** All 1,568 entity adjudication judgements were made by
hand by two of us, and they are the ground truth every extraction number rests on.
The scope decisions, the network analysis, the evaluation question set, and the
argument of the paper are ours. We verified AI-written code by reading it, by the
test suite, and by checking outputs against the source documents. Where we
disagreed with a suggestion we did not take it, and where results were uncertain we
said so in the report rather than smoothing them over.

We think the honest summary is that AI raised our ceiling rather than replacing our
work. It let us build something more ambitious than we could have alone, and it
made specific technical contributions we can point to. It did not decide what the
project was about, what counted as evidence, or what the results mean.

## Shared workspace

Google Drive: https://drive.google.com/drive/folders/1OZ3N_D0PXPqa8VXhwpbZmzQ6Jt5ePGDd
