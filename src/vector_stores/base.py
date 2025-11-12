"""
Base class for vector stores
All vector store implementations must inherit from this
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """
    Represents a document with content and metadata.
    """
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    All vector store implementations (Chroma, Weaviate, FAISS)
    must inherit from this class and implement the required methods.
    """
    
    def __init__(self, collection_name: str, embedding_function: Any = None):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection/index
            embedding_function: Function to generate embeddings
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
    
    @abstractmethod
    def add_documents(
        self, 
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List of document IDs
            
        Example:
            doc = Document(content="Hello", metadata={"source": "test"})
            ids = store.add_documents([doc])
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of most similar documents
            
        Example:
            results = store.similarity_search("What is RAG?", k=3)
        """
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples (document, similarity_score)
            
        Example:
            results = store.similarity_search_with_score("What is RAG?", k=3)
            # Returns: [(doc1, 0.95), (doc2, 0.87), (doc3, 0.82)]
        """
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        
        Example:
            store.delete_collection()
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
            
        Example:
            stats = store.get_collection_stats()
            # Returns: {"count": 100, "dimension": 1536}
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the vector store."""
        return f"{self.__class__.__name__}(collection='{self.collection_name}')"