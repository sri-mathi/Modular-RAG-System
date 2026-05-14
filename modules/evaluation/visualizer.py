import sys
import os
# Add the project root to sys.path so we can run this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from modules.vectorstores.chroma_store import ChromaStore
from core.base_embedder import BaseEmbedder

class EmbeddingVisualizer:
    def __init__(self, store: ChromaStore):
        self.store = store

    def visualize_2d(self, n_clusters=5, title="Modular RAG: Vector Space Analysis"):
        """Reduces embeddings to 2D using PCA and clusters them with K-Means."""
        # 1. Load the store and get data
        vector_store = self.store.load_store()
        print(f"DEBUG: Looking for collection '{self.store.collection_name}'...")
        
        data = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
        embeddings = data.get('embeddings', [])
        documents = data.get('documents', [])
        
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings found in the vector store. Have you ingested any data?")
            return

        print(f"Analyzing {len(embeddings)} embeddings...")

        # 2. Perform K-Means Clustering
        n_clusters = min(n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(embeddings)

        # 3. Perform PCA for 2D visualization
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # 4. Professional Plotting with Seaborn
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(16, 10))
        
        # Create the scatter plot with clusters as colors
        scatter = sns.scatterplot(
            x=reduced_embeddings[:, 0], 
            y=reduced_embeddings[:, 1], 
            hue=clusters, 
            palette="vlag", # Vibrant color palette
            s=150, 
            alpha=0.8, 
            edgecolor='w',
            legend="full"
        )

        # 5. Add intelligent labels (only for some points to avoid clutter)
        for i, doc in enumerate(documents):
            # Show labels for every 2nd point or based on some logic to keep it clean
            if i % 2 == 0:
                plt.annotate(
                    doc[:40].replace("\n", " ") + "...", 
                    (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                    fontsize=9, 
                    alpha=0.6,
                    xytext=(5, 5),
                    textcoords='offset points'
                )

        plt.title(title, fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Semantic Dimension 1", fontsize=12)
        plt.ylabel("Semantic Dimension 2", fontsize=12)
        plt.legend(title="Topic Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = "embedding_analysis_colorful.png"
        plt.savefig(output_path, dpi=300)
        print(f"Enhanced visualization saved to {output_path}")
        plt.show()

if __name__ == "__main__":
    from modules.embeddings.ollama_embedder import OllamaEmbedder
    embedder = OllamaEmbedder()
    store = ChromaStore(embedder=embedder)
    
    visualizer = EmbeddingVisualizer(store)
    visualizer.visualize_2d()
