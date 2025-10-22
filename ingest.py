import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import OPENAI_API_KEY, EMBED_MODEL, VECTOR_DIR, DOCS_DIR, require_api_key

def load_docs():
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "**", "*"), recursive=True):
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(path)
        elif ext in [".txt"]:
            loader = TextLoader(path, encoding="utf-8")
        elif ext in [".pdf"]:
            loader = PyPDFLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def main():
    require_api_key()
    os.makedirs(VECTOR_DIR, exist_ok=True)
    raw_docs = load_docs()
    if not raw_docs:
        print("No documents found in data/docs")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBED_MODEL)
    vector = FAISS.from_documents(chunks, embeddings)
    vector.save_local(VECTOR_DIR)
    print(f"Saved FAISS index to {VECTOR_DIR} with {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
