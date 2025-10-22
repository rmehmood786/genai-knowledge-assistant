import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
VECTOR_DIR = "data/vectorstore"
DOCS_DIR = "data/docs"

def require_api_key():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Create a .env file or export the variable.")
