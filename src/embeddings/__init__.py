"""
Embedding models package
Supports multiple embedding providers
"""

from .base import BaseEmbedding
from .openai_embeddings import OpenAIEmbedding
from .sentence_transformer import SentenceTransformerEmbedding

__all__ = [
    "BaseEmbedding",
    "OpenAIEmbedding", 
    "SentenceTransformerEmbedding"
]
