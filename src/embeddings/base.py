"""
Base class for embedding models
All embedding implementations must inherit from this
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding models.
    
    All embedding implementations (OpenAI, Sentence Transformers, etc.)
    must inherit from this class and implement the required methods.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name/identifier of the embedding model
        """
        self.model_name = model_name
        self.dimension = None  # Will be set by implementations
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
            
        Example:
            embeddings = model.embed_documents(["Hello world", "Test document"])
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
            
        Example:
            embedding = model.embed_query("What is RAG?")
            # Returns: [0.1, 0.2, 0.3, ...]
        """
        pass
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of embedding vectors
        """
        if self.dimension is None:
            # Get dimension by embedding a test string
            test_embedding = self.embed_query("test")
            self.dimension = len(test_embedding)
        return self.dimension
    
    def __repr__(self) -> str:
        """String representation of the embedding model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"