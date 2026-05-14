import os
from langchain_ollama import OllamaEmbeddings
from core.base_embedder import BaseEmbedder
from dotenv import load_dotenv

load_dotenv()

class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def get_embeddings(self) -> OllamaEmbeddings:
        print(f"Using Ollama Embeddings: {self.model_name}")
        return OllamaEmbeddings(
            model=self.model_name,
            base_url=self.base_url
        )
