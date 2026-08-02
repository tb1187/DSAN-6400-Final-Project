# DSAN-6400 Final Project — Network Analysis and Graph RAG for Document-Based Relationship Exploration

## Project Structure

### entity_extraction

### ingestion

### knowledge_graph

### retrieval
- `embeddings.py`: Defines a Python class to vector embed passages or a query using BAAI/bge-small-en-v1.5 and the sentence_transformers library.
- `index.py`: Defines helper functions that build/save a FAISS index of the vector embeddings for the passages, then loads them from the saved index for retrieval.
- `retriever.py`: Defines a Python class that takes a query, embeds it, and retrieves the top k most relevant passages from the FAISS index.
- `rag.py`: Creates a Python class that implements the full RAG pipeline, including vector store loading, Retriever definition (query embedding and top k document retrieval), and LLM calling.

## Setup


## Usage

### Retrieval Augmented Generation

1. Build the chunks: `python scripts/build_chunks.py`
2. Build the embeddings: `python scripts/build_index.py`
- Saves the embeddings to `data/processed/chunk_index.faiss`
3. Set `ANTHROPIC_API_KEY` in your environment
4. Query the pipeline:
   ```python
   from src.retrieval.rag import RAGPipeline
   RAGPipeline().answer("your question here")
   ```

## Team

Tyler Blue

Ethan Wotring

## Shared Workspaces
Google Drive: https://drive.google.com/drive/folders/1OZ3N_D0PXPqa8VXhwpbZmzQ6Jt5ePGDd

