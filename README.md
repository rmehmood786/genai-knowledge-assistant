# GenAI Knowledge Assistant

A small Retrieval-Augmented Generation (RAG) app that lets you chat with your internal knowledge using OpenAI embeddings and a local FAISS vector store. Built with Python, LangChain, and Streamlit.

## Features
- Document ingestion from Markdown, PDF, and text files
- OpenAI text-embedding-3-small embeddings + FAISS vector store
- Retrieval QA over your private docs
- Simple Streamlit interface
- Dockerfile and tests
- No data leaves your machine besides calls to OpenAI APIs

## Quick start

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Set your API key
Create a `.env` file at the project root:
```
OPENAI_API_KEY=sk-...
```

### 4) Ingest documents
Place your Markdown, PDF, or text files under `data/docs/`, then run:
```bash
python ingest.py
```

### 5) Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 and ask questions about your docs.

## Project structure
```
genai-knowledge-assistant/
├─ app.py                 # Streamlit UI for chat
├─ ingest.py              # Build the FAISS vector store from docs
├─ config.py              # Settings and helpers
├─ requirements.txt
├─ README.md
├─ .env.example
├─ .gitignore
├─ data/
│  ├─ docs/               # Put your source documents here
│  └─ vectorstore/        # Persisted FAISS index
├─ tests/
│  └─ test_ingest.py
└─ Dockerfile
```

## Docker
```bash
docker build -t genai-knowledge-assistant .
docker run -p 8501:8501 --env OPENAI_API_KEY=$OPENAI_API_KEY genai-knowledge-assistant
```

## Notes
- This demo uses `text-embedding-3-small` for cost and speed. You can switch to `text-embedding-3-large` in `ingest.py` if you need higher quality.
- For PDFs we use unstructured loaders via LangChain; if you have issues on your platform, convert PDFs to text or Markdown for best results.
- The vector store is local under `data/vectorstore/`.

## Licence
MIT
