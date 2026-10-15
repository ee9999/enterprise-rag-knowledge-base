# Enterprise RAG Knowledge Base

A document question-answering app built with **Python, OpenAI Embeddings, and Streamlit**. It retrieves relevant document passages and generates grounded answers with source references through a Traditional Chinese interface.

## Features
- Upload TXT, Markdown, or text-based PDF documents
- Chunk documents and generate OpenAI embeddings (`text-embedding-3-small`)
- Retrieve passages using cosine similarity and configurable Top-K
- Generate answers with `gpt-4.1-mini` and show cited source passages
- Return a no-answer message when retrieval finds no matching passages
- Basic error handling and automated tests

## Tech Stack
Python · OpenAI API · NumPy · Streamlit · PyMuPDF · pytest

## Project Structure
```text
enterprise-rag-knowledge-base/
├── app.py
├── rag/
│   ├── __init__.py
│   └── core.py
├── sample_data/
├── tests/
│   └── test_core.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start
```bash
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, or enter your key in the sidebar.

```bash
streamlit run app.py
```
Upload `sample_data/hr_policy.txt` and `sample_data/it_support.txt`, build the index, and ask a question in Chinese.

## Testing
```bash
python -m pytest -q
```
Tests use a fake API client and do not make paid API calls.

## Limitations
- Requires an OpenAI API key; API usage is billed separately.
- Scanned PDFs require OCR before ingestion; the index is held in session memory.
- Similarity thresholds are heuristic; model answers and citations require review.
- Use fictional or approved documents only. Uploaded content is sent to OpenAI.
