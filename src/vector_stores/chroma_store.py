"""
ChromaDB Vector Store Implementation
Primary vector database - explicitly mentioned in AJMS JD!
"""

import os
from typing import List, Dict, Any, Optional
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from .base import BaseVectorStore, Document


class ChromaStore(BaseVectorStore):
    """
    ChromaDB vector store implementation.
    
    Features:
    - In-memory or persistent storage
    - Fast similarity search
    - Metadata filtering
    - Easy to use, no external dependencies
    
    Perfect for development and production!
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_function: Any = None
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data (None for in-memory)
            embedding_function: Embedding function to use
            
        Example:
            # In-memory (for testing)
            store = ChromaStore(collection_name="my_docs")
            
            # Persistent (for production)
            store = ChromaStore(
                collection_name="my_docs",
                persist_directory="./chroma_db"
            )
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb not installed. "
                "Install with: pip install chromadb"
            )
        
        super().__init__(collection_name, embedding_function)
        
        # Initialize ChromaDB client
        if persist_directory:
            # Persistent storage
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            print(f"✓ ChromaDB initialized with persistence: {persist_directory}")
        else:
            # In-memory storage
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )
            print("✓ ChromaDB initialized (in-memory)")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document collection"}
        )
        
        print(f"✓ Collection ready: {collection_name}")
    
    def add_documents(
        self, 
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List of document IDs
            
        Example:
            doc = Document(content="RAG is cool", metadata={"source": "blog"})
            ids = store.add_documents([doc], embeddings=[[0.1, 0.2, ...]])
        """
        if not documents:
            return []
        
        # Generate IDs for documents
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Extract contents and metadata
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        if embeddings:
            # Use provided embeddings
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            # ChromaDB will generate embeddings (if embedding function provided)
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
        
        print(f"✓ Added {len(documents)} documents to {self.collection_name}")
        
        return ids
    
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
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter
        )
        
        # Convert to Document objects
        documents = []
        
        if results['documents'] and results['documents'][0]:
            for i, doc_content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                
                documents.append(Document(
                    content=doc_content,
                    metadata=metadata
                ))
        
        return documents
    
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
            for doc, score in results:
                print(f"Score: {score}, Content: {doc.content[:100]}")
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Convert to Document objects with scores
        documents_with_scores = []
        
        if results['documents'] and results['documents'][0]:
            for i, doc_content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0.0
                
                # Convert distance to similarity (lower distance = higher similarity)
                # Chroma uses L2 distance by default
                similarity = 1 / (1 + distance)
                
                doc = Document(
                    content=doc_content,
                    metadata=metadata
                )
                
                documents_with_scores.append((doc, similarity))
        
        return documents_with_scores
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        
        Example:
            store.delete_collection()
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"✓ Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"✗ Error deleting collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
            
        Example:
            stats = store.get_collection_stats()
            print(f"Documents: {stats['count']}")
        """
        count = self.collection.count()
        
        stats = {
            "collection_name": self.collection_name,
            "count": count,
            "metadata": self.collection.metadata
        }
        
        return stats
    
    def reset_collection(self) -> None:
        """
        Clear all documents from the collection without deleting it.
        
        Example:
            store.reset_collection()
        """
        # Get all IDs
        all_data = self.collection.get()
        
        if all_data['ids']:
            self.collection.delete(ids=all_data['ids'])
            print(f"✓ Reset collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} is already empty")