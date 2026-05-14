from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import json

class LLMReranker:
    def __init__(self, llm: BaseChatModel):
        """
        Initializes the LLM-based Re-ranker.
        
        Args:
            llm: The LLM (e.g., Groq) to use for scoring document relevance.
        """
        self.llm = llm

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        """
        Re-ranks a list of documents based on their relevance to the query.
        
        Args:
            query: The user query.
            documents: Initial list of retrieved documents.
            top_n: Number of documents to return after re-ranking.
        """
        if not documents:
            return []
            
        print(f"Re-ranking {len(documents)} documents using LLM...")
        
        # Prepare the list of documents for the LLM
        doc_texts = [f"ID {i}: {doc.page_content[:500]}..." for i, doc in enumerate(documents)]
        docs_str = "\n\n".join(doc_texts)
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert search evaluator. Given the following query and a list of retrieved document snippets, 
        identify the IDs of the top {top_n} most relevant documents that directly answer the query.
        
        Query: {query}
        
        Documents:
        {docs_str}
        
        Return ONLY a JSON list of the top {top_n} IDs in order of relevance, e.g., [2, 0, 5].
        Do not provide any explanation.
        """)
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "query": query,
                "docs_str": docs_str,
                "top_n": top_n
            })
            
            # Parse the response (handle potential formatting issues)
            content = response.content.strip()
            # Find the JSON part if the LLM added any text
            if "[" in content and "]" in content:
                content = content[content.find("["):content.rfind("]")+1]
                
            top_ids = json.loads(content)
            
            reranked_docs = []
            for doc_id in top_ids:
                if isinstance(doc_id, int) and 0 <= doc_id < len(documents):
                    reranked_docs.append(documents[doc_id])
            
            print(f"Successfully re-ranked and selected top {len(reranked_docs)} documents.")
            return reranked_docs[:top_n]
            
        except Exception as e:
            print(f"Error during re-ranking: {e}. Falling back to initial retrieval.")
            return documents[:top_n]

if __name__ == "__main__":
    print("LLMReranker module loaded.")
