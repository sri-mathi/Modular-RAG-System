import pandas as pd
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)

# Fix for different RAGAS versions
try:
    from ragas.metrics import answer_relevancy as answer_relevance
except ImportError:
    try:
        from ragas.metrics import answer_relevance
    except ImportError:
        # Fallback for even newer versions
        from ragas.metrics import AnswerRelevancy as answer_relevance

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from core.base_llm import BaseLLM
from core.base_embedder import BaseEmbedder

# Fix for Groq: answer_relevance defaults to n=3 variations which Groq rejects.
# Setting strictness=1 forces it to use n=1.
answer_relevance.strictness = 1

class RAGASEvaluator:
    def __init__(self, llm: BaseLLM, embedder: BaseEmbedder):
        """
        Initializes the RAGAS Evaluator using local Ollama models.
        """
        # RAGAS requires these specific wrappers for LangChain models
        self.evaluator_llm = LangchainLLMWrapper(llm.get_model())
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(embedder.get_embeddings())

    def evaluate_pipeline(self, 
                          questions: List[str], 
                          answers: List[str], 
                          contexts: List[List[str]], 
                          ground_truths: List[str] = None) -> pd.DataFrame:
        """
        Runs the RAGAS evaluation on the provided RAG outputs.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        
        # Define the metrics we want to measure
        # Faithfulness and Answer Relevance are reference-free
        metrics = [
            faithfulness,
            answer_relevance
        ]
        
        # Context Precision and Recall require a Ground Truth (reference)
        if ground_truths:
            metrics.append(context_precision)
            metrics.append(context_recall)

        print("--- Running RAGAS Evaluation (This may take a few minutes) ---")
        
        # Expert Config: High concurrency for fast API-based judging (Groq)
        run_config = RunConfig(timeout=600, max_workers=2)
        
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            run_config=run_config
        )
        
        # In newer RAGAS, the column names match the metric names exactly
        # We rename them here for consistency if needed
        df = result.to_pandas()
        if "answer_relevancy" in df.columns:
            df = df.rename(columns={"answer_relevancy": "answer_relevance"})
            
        return df
