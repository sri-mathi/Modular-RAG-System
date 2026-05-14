import os
import argparse
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from modules.ingestion.loader import DocumentIngestor
from modules.vectorstores.chroma_store import ChromaStore
from modules.generation.generator import Generator

# Import implementations
from modules.embeddings.ollama_embedder import OllamaEmbedder
from modules.generation.ollama_generator import OllamaLLM
from modules.retrieval.multi_query import MultiQueryManager

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Modular RAG System")
    parser.add_argument("--query", type=str, help="The question to ask the RAG system")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents from 'data' directory")
    parser.add_argument("--provider", type=str, default="groq", choices=["ollama", "groq", "openai"], help="Model provider")
    parser.add_argument("--advanced", action="store_true", help="Enable Multi-Query Expansion (Advanced Retrieval)")
    parser.add_argument("--graph", action="store_true", help="Enable Knowledge Graph Context (GraphRAG)")
    parser.add_argument("--build-graph", action="store_true", help="Extract entities and build the graph from documents")
    parser.add_argument("--visualize-graph", action="store_true", help="Generate an interactive HTML visualization of the knowledge graph")
    parser.add_argument("--rerank", action="store_true", help="Enable LLM-based Re-ranking (Cross-Encoder style)")
    args = parser.parse_args()

    # 1. Initialize modular components based on provider
    if args.provider == "groq":
        from modules.generation.groq_generator import GroqLLM
        embedder = OllamaEmbedder() 
        llm = GroqLLM() 
    elif args.provider == "ollama":
        embedder = OllamaEmbedder() 
        llm = OllamaLLM(model_name="llama3")
    else:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        print("OpenAI provider requested, but we are focusing on Groq/Ollama setup.")
        return

    ingestor = DocumentIngestor()
    store = ChromaStore(embedder=embedder)
    generator = Generator(llm=llm)
    
    # 2. Knowledge Graph Setup
    from modules.retrieval.graph_manager import GraphManager
    graph_manager = GraphManager(llm=llm.get_model())

    # 3. Ingestion & Graph Building
    if args.ingest or args.build_graph:
        if not os.path.exists("data"):
            os.makedirs("data")
            print("Created 'data' directory. Please add some PDF files there.")
            return
        
        raw_docs = ingestor.load_documents("data")
        if not raw_docs:
            print("No PDF files found in 'data' directory.")
            return
            
        chunks = ingestor.split_documents(raw_docs)
        
        if args.ingest:
            store.create_store(chunks)
        
        if args.build_graph:
            graph_manager.add_documents(chunks)
            
    if args.visualize_graph:
        graph_manager.visualize_graph()
        graph_manager.visualize_graph_png()
        return

    # 4. Retrieval & Generation
    if args.query:
        print(f"\n--- Question: {args.query} ---")
        
        # Setup retriever from vector store
        vector_store = store.load_store()
        
        # A. Vector Retrieval (Simple or Advanced)
        if args.advanced:
            print("\n--- Advanced Retrieval: Multi-Query Expansion ---")
            mq_manager = MultiQueryManager(llm=llm.get_model(), vector_store=vector_store)
            expanded = mq_manager.get_expanded_queries(args.query)
            print("Generated alternative queries:")
            for i, q in enumerate(expanded):
                print(f"  {i+1}. {q}")
            retriever = mq_manager.get_retriever()
        else:
            # If re-ranking is enabled, we fetch more documents initially (e.g., k=10)
            k = 10 if args.rerank else 3
            retriever = vector_store.as_retriever(search_kwargs={"k": k})

        # B. Re-ranking (Optional)
        if args.rerank:
            print("\n--- Re-ranking: LLM-based Filtering ---")
            initial_docs = retriever.invoke(args.query)
            from modules.retrieval.reranker import LLMReranker
            reranker = LLMReranker(llm=llm.get_model())
            docs = reranker.rerank(args.query, initial_docs)
            
            # Create a mock retriever that just returns these specific docs
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.callbacks import CallbackManagerForRetrieverRun
            
            class FixedRetriever(BaseRetriever):
                documents: List[Document]
                def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
                    return self.documents
            
            retriever = FixedRetriever(documents=docs)

        # C. Graph Retrieval (Optional)
        graph_context = ""
        if args.graph:
            print("\n--- Graph Retrieval: Traversing Relationships ---")
            graph_context = graph_manager.get_related_context(args.query)
            if graph_context:
                print("Found relevant graph relationships.")

        # C. Generation
        rag_chain = generator.get_chain(retriever)
        
        # If we have graph context, we'll append it to the query for the chain
        final_query = args.query
        if graph_context:
            final_query = f"{args.query}\n\nAdditional Knowledge Graph Facts:\n{graph_context}"

        response = rag_chain.invoke(final_query)
        print(f"\n--- Answer ---\n{response}")
    elif not args.ingest:
        print("Please provide a --query or use --ingest to load documents.")

if __name__ == "__main__":
    main()
