import os
from dotenv import load_dotenv
from modules.embeddings.ollama_embedder import OllamaEmbedder
from modules.generation.ollama_generator import OllamaLLM
from modules.generation.groq_generator import GroqLLM
from modules.vectorstores.chroma_store import ChromaStore
from modules.generation.generator import Generator
from modules.evaluation.evaluator import RAGASEvaluator

load_dotenv()

def main():
    # 1. Setup our RAG system
    embedder = OllamaEmbedder()
    llm = GroqLLM() # Now using llama-3.1-8b-instant for ultra-fast generation
    store = ChromaStore(embedder=embedder)
    generator = Generator(llm=llm)
    
    vector_store = store.load_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    rag_chain = generator.get_chain(retriever)
    
    # 2. Define our "Golden Questions"
    test_queries = [
        "What is the main architecture proposed in the paper?",
        "What are the benefits of using multi-head attention?",
        "How is the Scaled Dot-Product Attention calculated?",
        "What is the role of positional encodings in Transformers?"
    ]
    
    # 3. Collect RAG responses
    print(f"Generating answers for {len(test_queries)} evaluation questions...")
    
    questions = []
    answers = []
    contexts = []
    
    for query in test_queries:
        print(f"Processing: {query}")
        # Get context (manually for RAGAS)
        docs = retriever.invoke(query)
        context_list = [doc.page_content for doc in docs]
        
        # Get answer
        response = rag_chain.invoke(query)
        
        questions.append(query)
        answers.append(response)
        contexts.append(context_list)
        
    # 4. Run RAGAS Evaluation
    # Use Groq for the Judge (100x faster)
    judge_llm = GroqLLM()
    evaluator = RAGASEvaluator(llm=judge_llm, embedder=embedder)
    results_df = evaluator.evaluate_pipeline(
        questions=questions,
        answers=answers,
        contexts=contexts
    )
    
    # 5. Show results
    print("\n--- RAGAS Evaluation Results ---")
    available_columns = ['question', 'faithfulness', 'answer_relevance']
    cols_to_show = [c for c in available_columns if c in results_df.columns]
    print(results_df[cols_to_show])
    
    # Save to CSV for the DEVELOPMENT_LOG
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
