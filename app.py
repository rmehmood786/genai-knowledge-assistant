import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, EMBED_MODEL, VECTOR_DIR, require_api_key

st.set_page_config(page_title="GenAI Knowledge Assistant", layout="wide")
st.title("GenAI Knowledge Assistant")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY or "", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    model_name = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    st.markdown("Use the ingest script to add or update your documents.")
    ready = st.button("Load vector store")

require_api_key()

def load_vector():
    try:
        embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY), model=EMBED_MODEL)
        vector = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        return vector
    except Exception as e:
        st.error(f"Could not load vector store. Run `python ingest.py` first. Error: {e}")
        return None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if st.sidebar.button("Refresh vector store") or st.sidebar.checkbox("Auto-load", value=True):
    vec = load_vector()
    if vec:
        st.session_state.retriever = vec.as_retriever(search_kwargs={"k": 5})
        st.sidebar.success("Vector store loaded")

template = """You are a concise assistant that answers using only the provided context.
If the answer is not in the context, say you do not know.

Question:
{question}

Context:
{context}
"""
prompt = PromptTemplate.from_template(template)

if st.session_state.retriever:
    llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY), model_name=model_name, temperature=0.1)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.retriever, chain_type="stuff",
                                        chain_type_kwargs={"prompt": prompt})
    user_q = st.text_input("Ask a question about your documents")
    if user_q:
        with st.spinner("Thinking..."):
            result = chain.invoke({"query": user_q})
            st.write(result["result"])
else:
    st.info("Load the vector store to start. Place documents in data/docs and run `python ingest.py`.")

st.caption("Built with LangChain, OpenAI, FAISS, and Streamlit.")
