"""
Sentence Transformer Embeddings Implementation
Uses open-source sentence-transformers library (FREE!)
"""

from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Sentence Transformer embedding implementation.
    
    Uses open-source models from HuggingFace - completely FREE!
    
    Popular models:
    - all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
    - all-mpnet-base-v2 (768 dimensions, better quality, slower)
    - all-MiniLM-L12-v2 (384 dimensions, balanced)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize Sentence Transformer embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda', 'cpu', or None for auto)
            
        Example:
            # Fast and efficient (recommended for starting)
            embeddings = SentenceTransformerEmbedding(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Higher quality but slower
            embeddings = SentenceTransformerEmbedding(
                model_name="all-mpnet-base-v2"
            )
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        super().__init__(model_name=model_name)
        
        # Load model
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✓ Model loaded: {model_name}")
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
    
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
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            return embeddings_list
        
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
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            return embedding_list
        
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "device": str(self.model.device),
            "cost": "FREE (open-source)"
        }