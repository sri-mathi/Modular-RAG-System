from abc import ABC, abstractmethod
from typing import List
from langchain_core.embeddings import Embeddings

class BaseEmbedder(ABC):
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Returns the LangChain compatible Embeddings object."""
        pass
