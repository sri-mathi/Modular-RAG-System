from typing import List
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    try:
        from langchain_community.retrievers import MultiQueryRetriever
    except ImportError:
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class LineList(BaseModel):
    # "lines" is the key (list of strings) that the parser will look for
    lines: List[str] = Field(description="Lines of text")

class MultiQueryManager:
    def __init__(self, llm: BaseChatModel, vector_store: VectorStore):
        """
        Initializes the Multi-Query Retriever.
        
        Args:
            llm: A LangChain compatible ChatModel (e.g., OllamaLLM)
            vector_store: The vector store to search against
        """
        self.llm = llm
        self.vector_store = vector_store

    def get_retriever(self):
        """
        Creates and returns a MultiQueryRetriever instance.
        """
        return MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),
            llm=self.llm
        )

    def get_expanded_queries(self, question: str) -> List[str]:
        """
        Expert level: Manually generate and show the expanded queries.
        This is useful for understanding what the LLM is thinking.
        """
        prompt = f"""You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}"""
        
        response = self.llm.invoke(prompt)
        # Handle both string and Message objects
        content = response.content if hasattr(response, 'content') else str(response)
        
        queries = [q.strip() for q in content.split("\n") if q.strip()]
        
        # Expert Filter: Remove "polite" intro sentences like "Here are the versions..."
        clean_queries = []
        for q in queries:
            # Skip if it doesn't end with a question mark or looks like an intro
            low_q = q.lower()
            if "here are" in low_q or "different versions" in low_q or ":" in q:
                continue
            if len(q) > 10:
                clean_queries.append(q)
        
        return clean_queries[:3]
