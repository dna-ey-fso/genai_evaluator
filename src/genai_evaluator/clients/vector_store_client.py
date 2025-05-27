import pickle
from pathlib import Path
from typing import List, Union

import chromadb
import faiss
import numpy as np
from chromadb.config import Settings
from loguru import logger
from PyPDF2 import PdfReader

from genai_evaluator.clients.embedding_client import EmbeddingClient
from genai_evaluator.clients.utils import extract_text_from_pdf
from genai_evaluator.interfaces.interfaces import (
    EmbeddingClient,
    Retrieval,
    VectorStoreClient,
)


class FAISSClient(VectorStoreClient):
    """
    FAISS-based vector store for document retrieval.
    """

    def __init__(self, embedding_client: EmbeddingClient, dimension: int = 1536):
        self.embedding_client = embedding_client
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add_documents(self, documents: List[str], batch_size: int = 32) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of text documents to index
            batch_size: Number of documents to process at once
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            embeddings = [self.embedding_client.embed(doc) for doc in batch]

            embeddings_array = np.array(embeddings).astype("float32")

            self.index.add(embeddings_array)

            self.documents.extend(batch)

    def ingest_documents(
        self, source: Union[List[str], str, Path], batch_size: int = 32
    ) -> None:
        """
        Unified method to ingest documents from multiple sources:
        - List of strings (document content)
        - Single file path (.txt or .pdf)
        - Directory path (containing .txt or .pdf files)

        Args:
            source: Either a list of document strings, a file path, or directory path
            batch_size: Number of documents to process at once
        """
        documents = []

        # Case 1: List of strings provided
        if isinstance(source, list):
            documents = source

        # Case 2: Path provided (file or directory)
        elif isinstance(source, (str, Path)):
            path = Path(source)

            # Single file
            if path.is_file():
                if path.suffix.lower() == ".txt":
                    with open(path, "r", encoding="utf-8") as f:
                        documents.append(f.read())
                    logger.debug(f"Ingested text from file: {path}")

                elif path.suffix.lower() == ".pdf":
                    pdf_documents = extract_text_from_pdf(path)
                    if pdf_documents:
                        documents.extend(pdf_documents)

            # Directory of files
            elif path.is_dir():
                # Process text files
                for file_path in path.glob("**/*.txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        documents.append(f.read())

                # Process PDF files
                for file_path in path.glob("**/*.pdf"):
                    pdf_documents = extract_text_from_pdf(file_path)
                    if pdf_documents:
                        documents.extend(pdf_documents)

        # Add all documents to the vector store
        if documents:
            self.add_documents(documents, batch_size)
            logger.info(f"Total documents ingested: {len(documents)}")
        else:
            logger.warning("No documents were ingested")

    def search(self, query: str, k: int = 5) -> List[Retrieval]:
        """
        Search for similar documents based on a query.

        Args:
            query: Text query to search for
            k: Number of results to return

        Returns:
            List of retrieval results with content and search scores
        """
        # Get query embedding
        query_embedding = self.embedding_client.embed(query)
        query_embedding_array = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding_array, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx != -1:
                results.append(
                    Retrieval(
                        content=self.documents[idx],
                        search_score=float(1.0 / (1.0 + distances[0][i])),
                        rerank_score=None,
                    )
                )

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save the vector store to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path) + ".index")

        with open(str(path) + ".documents", "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(
        cls, path: Union[str, Path], embedding_client: EmbeddingClient
    ) -> "FAISSClient":
        """Load a vector store from disk"""
        path = Path(path)

        instance = cls(embedding_client)

        instance.index = faiss.read_index(str(path) + ".index")
        instance.dimension = instance.index.d

        with open(str(path) + ".documents", "rb") as f:
            instance.documents = pickle.load(f)

        return instance


class ChromaDBClient(VectorStoreClient):
    """
    ChromaDB-based vector store for document retrieval.
    """

    def __init__(self, embedding_client: EmbeddingClient, path: str, collection_name: str):
        """
        Initialize the ChromaDB client.
        """
        self.embedding_client = embedding_client
        self.collection = self._initialize_chromadb(path, collection_name)


    def _initialize_chromadb(self, path: str, collection_name: str):
        """
        Initialize the ChromaDB collection.
        """

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        client = chromadb.PersistentClient(path=str(path))

        # Create or get collection
        try:
            collection = client.get_collection(collection_name)
            logger.info(f"Using existing ChromaDB collection: {collection_name}")
        except Exception:    
            logger.info(f"Collection {collection_name} not found.")

        return collection

    def add_documents(self, documents: List[str], batch_size: int = 32) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of text documents to index
            batch_size: Number of documents to process at once
        """
        # Implementation for adding documents to ChromaDB
        pass

    def search(self, query: str, k: int = 5) -> List[Retrieval]:
        """
        Search for similar documents based on a query.

        Args:
            query: Text query to search for
            k: Number of results to return

        Returns:
            List of retrieval results with content and search scores
        """
        # Implementation for searching in ChromaDB
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Save the vector store to disk"""
        # Implementation for saving ChromaDB vector store
        pass

    @classmethod
    def load(
        cls, path: Union[str, Path], embedding_client: EmbeddingClient
    ) -> "ChromaDBClient":
        """Load a vector store from disk"""
        # Implementation for loading ChromaDB vector store
        pass
