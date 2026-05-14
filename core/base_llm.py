from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel

class BaseLLM(ABC):
    @abstractmethod
    def get_model(self) -> BaseChatModel:
        """Returns the LangChain compatible ChatModel object."""
        pass
