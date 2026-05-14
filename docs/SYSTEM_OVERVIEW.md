# Expert-Grade Modular RAG: System Overview & Architecture

Welcome to the **Modular RAG Pipeline**. This system has been engineered to transition from a basic retrieval script into a high-performance, production-ready architecture.

---

## 1. System Architecture
The system is built on **Modular Abstraction**. Every major component (LLM, Embedder, Vector Store) is decoupled using Abstract Base Classes (ABCs). This allows you to swap providers (e.g., move from Ollama to OpenAI) by changing a single line of code.

### **The "Brain" (LLM)**
- **Primary Provider**: **Groq** (`llama-3.1-8b-instant`).
- **Use Cases**: Generation, Judging (RAGAS), Entity Extraction (GraphRAG), and Re-ranking.
- **Backup Provider**: **Ollama** (`llama3`).

### **The "Memory" (Embeddings & Storage)**
- **Embedder**: **nomic-embed-text** (via Ollama).
- **Vector Database**: **ChromaDB**.
- **Distance Metric**: **Cosine Similarity** (Scale-invariant semantic matching).

---

## 2. Advanced Retrieval Pipeline
This is what makes the system "Expert-Grade." It uses a multi-stage process to ensure the highest precision:

### **Stage A: Multi-Query Expansion**
The system uses an LLM to generate 3-5 mathematical and conceptual variations of the user's query to ensure we catch all relevant documents.

### **Stage B: Hybrid Retrieval (GraphRAG)**
In addition to vector similarity, the system traverses a **Knowledge Graph** (built using NetworkX). It identifies entities in the query and finds their structural relationships (e.g., "Attention" -> "Transformer").

### **Stage C: LLM Re-ranking**
The system retrieves the top 10 candidates from the vector store and uses the Groq LLM as a "Cross-Encoder" to select only the top 3 absolute best matches.

---

## 3. Evaluation & Observability
### **RAGAS Framework**
The system includes an automated evaluation suite (`evaluate_rag.py`) that measures:
- **Faithfulness**: Is the answer derived from the retrieved documents?
- **Answer Relevance**: Does the answer directly address the user's query?

### **Visual Analytics**
- **Embedding Clusters**: PCA-based 2D visualization of how your documents are clustered in the vector space.
- **Knowledge Maps**: Interactive HTML and static PNG visualizations of the relationships extracted from your data.

---

## 4. How to Use the System

### **A. Core Commands**
- **Ingest Documents**: Load PDFs into the vector store.
  ```bash
  python3 simple_rag.py --ingest
  ```
- **Standard Query**: Simple vector search.
  ```bash
  python3 simple_rag.py --query "Your question"
  ```
- **Expert Query**: Hybrid search with re-ranking.
  ```bash
  python3 simple_rag.py --query "Your question" --graph --rerank
  ```

### **B. Advanced Visualization**
- **Visualize Vector Space**:
  ```bash
  python3 visualizer.py
  ```
- **Visualize Knowledge Graph**:
  ```bash
  python3 simple_rag.py --visualize-graph
  ```

### **C. Evaluation**
- **Run Performance Benchmarks**:
  ```bash
  python3 evaluate_rag.py
  ```

---

## 5. Technical Stack
- **Orchestration**: LangChain
- **LLM API**: Groq
- **Local Models**: Ollama
- **Graph Logic**: NetworkX & PyVis
- **Evaluation**: RAGAS & Pandas
- **Visualization**: Matplotlib, Seaborn, Scikit-learn
