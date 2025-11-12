"""
Vector store package
Supports multiple vector database backends
"""

from .base import BaseVectorStore, Document
from .chroma_store import ChromaStore
from .weaviate_store import WeaviateStore

__all__ = [
    "BaseVectorStore",
    "Document",
    "ChromaStore",
    "WeaviateStore"
]