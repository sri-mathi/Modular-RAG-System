import streamlit as st
import os
import time
from typing import List
from dotenv import load_dotenv

# Core RAG Modules
from modules.embeddings.ollama_embedder import OllamaEmbedder
from modules.generation.groq_generator import GroqLLM
from modules.vectorstores.chroma_store import ChromaStore
from modules.ingestion.loader import DocumentIngestor
from modules.generation.generator import Generator
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# --- Page Config ---
st.set_page_config(page_title="RAG Chat", page_icon="⚡", layout="centered")
load_dotenv()

# --- Custom BaseRetriever ---
class FixedRetriever(BaseRetriever):
    documents: List[Document]
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        return self.documents

# --- 1. State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    embedder = OllamaEmbedder()
    llm = GroqLLM()
    st.session_state.engine = {
        "embedder": embedder,
        "llm": llm,
        "store": ChromaStore(embedder=embedder),
        "generator": Generator(llm=llm)
    }

engine = st.session_state.engine

# --- 2. Minimal UI Layout ---
# Sticky Header CSS
st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:has(div.sticky-header) {
        position: sticky;
        top: 2.8rem;
        background-color: white;
        z-index: 999;
        padding-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

header_container = st.container()
with header_container:
    st.markdown('<div class="sticky-header"></div>', unsafe_allow_html=True)
    st.title("Modular RAG System")
    st.markdown("Upload a document and start talking to it instantly.")

    # Upload Section
    uploaded_file = st.file_uploader("", type=["pdf", "docx"])

    if uploaded_file:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            with st.status(f"Ingesting {uploaded_file.name}...", expanded=True) as status:
                # 1. Clear old data to ensure we only chat with the NEW file
                if not os.path.exists("./data"): 
                    os.makedirs("./data")
                else:
                    for f in os.listdir("./data"):
                        try: os.remove(os.path.join("./data", f))
                        except: pass
                
                # 2. Save new file
                file_path = os.path.join("./data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 3. Process
                ingestor = DocumentIngestor()
                raw_docs = ingestor.load_documents("./data")
                chunks = ingestor.split_documents(raw_docs)
                engine["store"].create_store(chunks)
                
                st.session_state.last_uploaded = uploaded_file.name
                status.update(label="Knowledge Base Ready!", state="complete")
                st.toast("New Document Ingested (Previous Data Cleared)")

# --- 3. Chat Loop ---
# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Retrieval
        store = engine["store"].load_store()
        retriever = store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(prompt)
        
        # Generation
        rag_chain = engine["generator"].get_chain(FixedRetriever(documents=docs))
        response = rag_chain.invoke(prompt)
        
        response_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
