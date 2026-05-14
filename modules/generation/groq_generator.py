import os
from langchain_groq import ChatGroq
from core.base_llm import BaseLLM
from dotenv import load_dotenv

load_dotenv()

class GroqLLM(BaseLLM):
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

    def get_model(self) -> ChatGroq:
        print(f"Using Groq LLM: {self.model_name}")
        return ChatGroq(
            model_name=self.model_name,
            groq_api_key=self.api_key,
            temperature=0,
            n=1
        )
