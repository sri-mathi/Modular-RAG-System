import os
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.base_embedder import BaseEmbedder

load_dotenv()

class ChromaStore:
    def __init__(self, embedder: BaseEmbedder, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = embedder.get_embeddings()
        self.vector_store = None

    def create_store(self, documents: List[Document]):
        """Creates a new vector store from documents."""
        print(f"Creating vector store in {self.persist_directory}...")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"} # Use Cosine Similarity
        )
        print("Vector store created and persisted.")
        return self.vector_store

    def load_store(self):
        """Loads an existing vector store."""
        print(f"Loading vector store from {self.persist_directory}...")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return self.vector_store

    def search(self, query: str, k: int = 4):
        """Simple similarity search."""
        if not self.vector_store:
            self.load_store()
        return self.vector_store.similarity_search(query, k=k)

if __name__ == "__main__":
    # Test initialization (requires an embedder now)
    # from modules.embeddings.ollama_embedder import OllamaEmbedder
    # embedder = OllamaEmbedder()
    # store = ChromaStore(embedder=embedder)
    print("ChromaStore module loaded.")
