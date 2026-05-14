import networkx as nx
import os
import json
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

class GraphManager:
    def __init__(self, llm: BaseChatModel, graph_path: str = "./graph_db.json"):
        """
        Initializes the GraphRAG Manager.
        
        Args:
            llm: The LLM to use for entity and relationship extraction.
            graph_path: Path to save/load the NetworkX graph.
        """
        self.llm = llm
        self.graph_path = graph_path
        self.graph = nx.Graph()
        
        # Initialize the LangChain Graph Transformer
        # This is a powerful experimental tool that handles extraction
        self.transformer = LLMGraphTransformer(llm=self.llm)
        
        if os.path.exists(self.graph_path):
            self.load_graph()

    def add_documents(self, documents: List[Document]):
        """
        Extracts triples from documents and adds them to the graph.
        Processes in small batches with delays to respect API rate limits.
        """
        import time
        batch_size = 5
        print(f"Extracting graph data from {len(documents)} document chunks in batches of {batch_size}...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
            
            try:
                graph_documents = self.transformer.convert_to_graph_documents(batch)
                
                for graph_doc in graph_documents:
                    # Add Nodes
                    for node in graph_doc.nodes:
                        self.graph.add_node(node.id, type=node.type)
                    
                    # Add Edges (Relationships)
                    for edge in graph_doc.relationships:
                        self.graph.add_edge(
                            edge.source.id, 
                            edge.target.id, 
                            relation=edge.type
                        )
                
                self.save_graph()
                # Pause to avoid RateLimitError on Groq free tier
                print("Sleeping for 2 seconds to respect rate limits...")
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}. Skipping this batch.")
                continue
        
        print(f"Graph update complete: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def get_related_context(self, query: str, depth: int = 1) -> str:
        """
        Identifies entities in the query using the LLM and finds their neighborhood in the graph.
        """
        # 1. LLM-based Entity Extraction from Query
        # We ask the LLM to identify keywords from the query that might be in our graph nodes.
        prompt = f"Identify the top 3-5 key entities or concepts in this query: '{query}'. Return only a comma-separated list of entities."
        try:
            response = self.llm.invoke(prompt)
            entities_str = response.content
            # Clean up the response
            query_entities = [e.strip().strip("'\"") for e in entities_str.split(",")]
            print(f"LLM identified entities for graph lookup: {query_entities}")
        except Exception as e:
            print(f"Error during LLM entity extraction: {e}. Falling back to keyword match.")
            query_entities = [node for node in self.graph.nodes if node.lower() in query.lower()]

        if not query_entities:
            return ""

        # 2. Traverse Graph with Fuzzy Matching
        # We look for nodes that are similar to our extracted entities
        subgraph_nodes = set()
        for extracted_entity in query_entities:
            # Find the closest match in our graph nodes
            best_match = None
            for node in self.graph.nodes:
                if extracted_entity.lower() in node.lower() or node.lower() in extracted_entity.lower():
                    best_match = node
                    break
            
            if best_match:
                try:
                    neighborhood = nx.single_source_shortest_path_length(self.graph, best_match, cutoff=depth)
                    subgraph_nodes.update(neighborhood.keys())
                except nx.NetworkXError:
                    continue
        
        # 3. Format Context
        relevant_edges = []
        subgraph = self.graph.subgraph(subgraph_nodes)
        for u, v, d in subgraph.edges(data=True):
            relevant_edges.append(f"({u}) --[{d.get('relation', 'related to')}]--> ({v})")
        
        if not relevant_edges:
            return ""
            
        context = "\n--- Graph Knowledge ---\n"
        context += "\n".join(relevant_edges)
        return context

    def visualize_graph_png(self, output_path: str = "knowledge_graph.png"):
        """
        Generates a static PNG visualization of the graph using Matplotlib.
        """
        import matplotlib.pyplot as plt
        print(f"Generating static image in {output_path}...")
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph, k=0.15, iterations=20)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color="skyblue")
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family="sans-serif")
        
        plt.title("Knowledge Graph Structure")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Static visualization saved to {output_path}")

    def visualize_graph(self, output_path: str = "knowledge_graph.html"):
        """
        Generates an interactive HTML visualization of the graph using PyVis.
        """
        from pyvis.network import Network
        print(f"Generating visualization in {output_path}...")
        
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
        
        # Add nodes with colors based on their type
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'Entity')
            net.add_node(node, label=node, title=f"Type: {node_type}")
            
        # Add edges
        for u, v, d in self.graph.edges(data=True):
            net.add_edge(u, v, title=d.get('relation', 'related to'))
            
        net.write_html(output_path)
        print(f"Visualization saved to {output_path}")

    def save_graph(self):
        """Saves the graph to a JSON file."""
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, 'w') as f:
            json.dump(data, f)
        print(f"Graph saved to {self.graph_path}")

    def load_graph(self):
        """Loads the graph from a JSON file."""
        try:
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
            print(f"Loaded existing graph with {self.graph.number_of_nodes()} nodes.")
        except Exception as e:
            print(f"Error loading graph: {e}. Starting fresh.")
            self.graph = nx.Graph()

if __name__ == "__main__":
    # Test initialization logic
    print("GraphManager module loaded.")
