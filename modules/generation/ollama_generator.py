import os
from langchain_ollama import ChatOllama
from core.base_llm import BaseLLM
from dotenv import load_dotenv

load_dotenv()

class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = "llama3", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def get_model(self) -> ChatOllama:
        print(f"Using Ollama LLM: {self.model_name}")
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url,
            timeout=300 # 5 minute timeout per request
        )
