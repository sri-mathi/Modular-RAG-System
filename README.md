# Modular RAG System: Expert-Grade AI Pipeline 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)
[![Ollama](https://img.shields.io/badge/Embedder-Ollama-lightgrey.svg)](https://ollama.com/)

## 🏗️ System Architecture
![System Architecture](architecture_diagram.png)

This is not just another RAG script. This is a **Modular, High-Performance RAG Architecture** designed for precision, speed, and structural intelligence.

## 🌟 Expert Features
- **Hybrid Retrieval (GraphRAG)**: Combines semantic vector search (ChromaDB) with structural relationship traversal (NetworkX).
- **LLM-Based Re-ranking**: Uses Groq as a Cross-Encoder to filter the top 10 candidates down to the 3 most relevant facts, eliminating noise.
- **Multi-Query Expansion**: Handles complex user queries by generating mathematical and conceptual variations.
- **Automated Evaluation (RAGAS)**: Built-in quantitative benchmarking for **Faithfulness** and **Relevance**.
- **High Performance**: Optimized using **Groq** for 50x faster generation compared to local execution.
- **Visual Analytics**: Interactive Knowledge Graph maps and Embedding Cluster visualizations (PCA).

### **Knowledge Graph Visualization**
![Knowledge Graph](knowledge_graph.png)

### **Embedding Cluster Analysis**
![Embedding Clusters](embedding_analysis_colorful.png)

## 📊 Performance Benchmarks
| Metric | Score | Insight |
| :--- | :--- | :--- |
| **Faithfulness** | **0.82** | High factual grounding; minimal hallucinations. |
| **Answer Relevance** | **1.00** | Perfect alignment with user intent. |

## 🛠️ Tech Stack
- **Orchestration**: LangChain
- **LLM API**: Groq (Llama 3.1 8B Instant)
- **Local Embeddings**: Ollama (nomic-embed-text)
- **Vector DB**: ChromaDB (Cosine Similarity)
- **Graph Logic**: NetworkX & PyVis
- **Evaluation**: RAGAS

## 🚀 Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/expert-rag-system
cd expert-rag-system

# Install dependencies
pip install -r requirements.txt

# Configure your .env
echo "GROQ_API_KEY=your_key_here" > .env
```

### 2. Ingest & Build Knowledge
```bash
# Ingest PDFs and build the vector database
python3 simple_rag.py --ingest

# Build the Knowledge Graph (GraphRAG)
python3 simple_rag.py --build-graph
```

### 3. Run Expert Query
```bash
python3 simple_rag.py --query "What is the Transformer architecture?" --graph --rerank
```

### 4. Visualize
```bash
# Generate Knowledge Map (HTML + PNG)
python3 simple_rag.py --visualize-graph

# Generate Embedding Clusters (PCA)
python3 visualizer.py
```

## 📂 Architecture
The system follows a modular design pattern:
- `modules/generation`: LLM provider abstraction.
- `modules/embeddings`: Embedding provider abstraction.
- `modules/retrieval`: Advanced logic (Graph, Multi-Query, Re-ranker).
- `modules/evaluation`: RAGAS benchmarking logic.

