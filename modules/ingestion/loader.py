import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentIngestor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents(self, directory_path: str) -> List[Document]:
        """Loads all PDFs from a directory."""
        print(f"Loading documents from {directory_path}...")
        loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages.")
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks."""
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        return chunks

if __name__ == "__main__":
    # Example usage
    ingestor = DocumentIngestor()
    # Assume 'data' directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add some PDF files there.")
    else:
        # This is just a test to see if it works
        # documents = ingestor.load_documents("data")
        # chunks = ingestor.split_documents(documents)
        pass
