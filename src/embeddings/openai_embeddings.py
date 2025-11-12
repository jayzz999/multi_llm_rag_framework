"""
OpenAI Embeddings Implementation
Uses OpenAI's embedding models
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import BaseEmbedding

# Load environment variables
load_dotenv()


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI embedding implementation.
    
    Supports models:
    - text-embedding-3-small (1536 dimensions, cheaper)
    - text-embedding-3-large (3072 dimensions, better quality)
    - text-embedding-ada-002 (legacy, 1536 dimensions)
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (if not in environment)
            
        Example:
            embeddings = OpenAIEmbedding(model_name="text-embedding-3-small")
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai"
            )
        
        super().__init__(model_name=model_name)
        
        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Set dimension based on model
        self.dimension = self._get_model_dimension()
    
    def _get_model_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model_name, 1536)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
            
        Example:
            embeddings = model.embed_documents(["Hello", "World"])
        """
        if not texts:
            return []
        
        # Remove empty texts
        texts = [text if text else " " for text in texts]
        
        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            return embeddings
        
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
            
        Example:
            embedding = model.embed_query("What is RAG?")
        """
        if not text:
            text = " "
        
        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            
            # Extract embedding
            embedding = response.data[0].embedding
            
            return embedding
        
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def get_cost_estimate(self, num_tokens: int) -> float:
        """
        Estimate cost for embedding given number of tokens.
        
        Args:
            num_tokens: Number of tokens to embed
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10
        }
        
        price_per_million = pricing.get(self.model_name, 0.02)
        cost = (num_tokens / 1_000_000) * price_per_million
        
        return cost