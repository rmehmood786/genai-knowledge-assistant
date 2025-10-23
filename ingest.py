import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import OPENAI_API_KEY, EMBED_MODEL, VECTOR_DIR, DOCS_DIR, require_api_key

SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf"}

def load_docs():
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "**", "*"), recursive=True):
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue

        try:
            if ext == ".pdf":
                # Use PyPDFLoader (self-contained, no extra downloads)
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            else:
                # For .md/.markdown/.txt use a simple text loader to avoid unstructured downloads
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue
    return docs

def main():
    require_api_key()
    os.makedirs(VECTOR_DIR, exist_ok=True)

    raw_docs = load_docs()
    if not raw_docs:
        print("No documents found in data/docs. Add .md, .txt or .pdf files and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vector = FAISS.from_documents(chunks, embeddings)
    vector.save_local(VECTOR_DIR)
    print(f"Saved FAISS index to {VECTOR_DIR} with {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
