import os
import streamlit as st

# Vector store + embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Prompt template (new import path for LC ≥0.2)
from langchain_core.prompts import PromptTemplate

# Optional OpenAI (if you later want cloud LLMs)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Local LLM via Transformers
from transformers import pipeline

from config import OPENAI_API_KEY, EMBED_MODEL, VECTOR_DIR, require_api_key

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="GenAI Knowledge Assistant", layout="wide")
st.title("GenAI Knowledge Assistant")

with st.sidebar:
    st.header("Settings")

    # Free local toggles
    use_free_embeddings = st.checkbox(
        "Use free local embeddings (SentenceTransformers)",
        value=True,
        help="Uncheck to use OpenAI embeddings (requires API billing)."
    )
    use_local_llm = st.checkbox(
        "Use free local LLM (Transformers)",
        value=True,
        help="Uncheck to use OpenAI chat models (requires API billing)."
    )

    # Only relevant if you turn local LLM OFF
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY or "", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model_name = st.selectbox(
        "OpenAI chat model (if local LLM is OFF)",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )

    st.markdown("Use `python ingest.py` to add or update your documents.")
    st.button("Refresh vector store", key="refresh")

# ---------------- Helpers ----------------
def get_embeddings():
    if use_free_embeddings:
        # Free, **offline** embeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Paid OpenAI embeddings (kept for completeness)
    return OpenAIEmbeddings(model=EMBED_MODEL)

class LocalLLM:
    """Tiny wrapper so local pipeline has a similar .invoke() interface."""
    def __init__(self, model_id="google/flan-t5-small"):
        # flan-t5-small is light and instruction-tuned (good for Q&A)
        self.pipe = pipeline("text2text-generation", model=model_id)
    def invoke(self, prompt: str):
        out = self.pipe(prompt, max_new_tokens=256, do_sample=False)
        # mimic ChatOpenAI result object with .content
        return type("Resp", (), {"content": out[0]["generated_text"]})

def get_llm():
    if use_local_llm:
        return LocalLLM("google/flan-t5-small")
    # Cloud LLM (needs billing + API key)
    return ChatOpenAI(model=model_name, temperature=0.1)

def load_vector():
    try:
        embeddings = get_embeddings()
        vector = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        return vector
    except Exception as e:
        st.error(f"Could not load vector store. Run `python ingest.py` first. Error: {e}")
        return None

def answer_with_rag(query: str, retriever, llm, prompt: PromptTemplate):
    # New LangChain retrievers are runnables → use .invoke()
    docs = retriever.invoke(query)
    if not docs:
        return "I could not find anything relevant in your documents."

    context = "\n\n".join(d.page_content[:2000] for d in docs)
    full_prompt = prompt.format(question=query, context=context)
    resp = llm.invoke(full_prompt)
    return getattr(resp, "content", str(resp))

# ---------------- Session state ----------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ---------------- Load vector store ----------------
if st.sidebar.checkbox("Auto-load vector store", value=True) or st.session_state.get("refresh"):
    vec = load_vector()
    if vec:
        st.session_state.retriever = vec.as_retriever(search_kwargs={"k": 5})
        st.sidebar.success("Vector store loaded successfully.")

# ---------------- Prompt ----------------
template = """You are a concise assistant that answers using only the provided context.
If the answer is not in the context, say you do not know.

Question:
{question}

Context:
{context}
"""
prompt = PromptTemplate.from_template(template)

# ---------------- Main UI ----------------
if st.session_state.retriever:
    llm = get_llm()
    user_q = st.text_input("Ask a question about your documents")
    if user_q:
        with st.spinner("Thinking..."):
            answer = answer_with_rag(user_q, st.session_state.retriever, llm, prompt)
            st.write(answer)
else:
    st.info("Load the vector store to start. Place documents in data/docs and run `python ingest.py`.")

st.caption("Built with FAISS, SentenceTransformers, Transformers, Streamlit, and optional OpenAI.")
