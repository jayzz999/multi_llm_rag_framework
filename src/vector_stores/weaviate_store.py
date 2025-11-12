"""
Weaviate Vector Store Implementation
For comparison and advanced features
"""

import os
from typing import List, Dict, Any, Optional
import uuid

try:
    import weaviate
    from weaviate.auth import AuthApiKey
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from dotenv import load_dotenv
from .base import BaseVectorStore, Document

load_dotenv()


class WeaviateStore(BaseVectorStore):
    """
    Weaviate vector store implementation.
    
    Features:
    - Cloud or self-hosted
    - Scalable for production
    - Advanced filtering
    - GraphQL API
    
    Can be used locally with Docker or in cloud.
    """
    
    def __init__(
        self,
        collection_name: str = "Document",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_function: Any = None
    ):
        """
        Initialize Weaviate vector store.
        
        Args:
            collection_name: Name of the class/collection
            url: Weaviate instance URL (default: from env or localhost)
            api_key: API key for cloud instance (optional)
            embedding_function: Embedding function to use
            
        Example:
            # Local Docker instance
            store = WeaviateStore(
                collection_name="Documents",
                url="http://localhost:8080"
            )
            
            # Cloud instance
            store = WeaviateStore(
                collection_name="Documents",
                url="https://your-cluster.weaviate.network",
                api_key="your-api-key"
            )
        """
        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "weaviate-client not installed. "
                "Install with: pip install weaviate-client"
            )
        
        super().__init__(collection_name, embedding_function)
        
        # Get URL and API key
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")
        
        # Initialize Weaviate client
        try:
            if self.api_key:
                # Cloud instance with authentication
                auth_config = AuthApiKey(api_key=self.api_key)
                self.client = weaviate.Client(
                    url=self.url,
                    auth_client_secret=auth_config
                )
            else:
                # Local instance without authentication
                self.client = weaviate.Client(url=self.url)
            
            print(f"✓ Connected to Weaviate: {self.url}")
        
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Weaviate at {self.url}. "
                f"Make sure Weaviate is running. Error: {str(e)}"
            )
        
        # Create schema if it doesn't exist
        self._ensure_schema_exists()
    
    def _ensure_schema_exists(self) -> None:
        """Create the schema/class if it doesn't exist."""
        # Check if class exists
        schema = self.client.schema.get()
        class_exists = any(
            c['class'] == self.collection_name 
            for c in schema.get('classes', [])
        )
        
        if not class_exists:
            # Define class schema
            class_schema = {
                "class": self.collection_name,
                "description": "Document collection for RAG",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Document metadata (JSON string)"
                    }
                ]
            }
            
            # Create class
            self.client.schema.create_class(class_schema)
            print(f"✓ Created Weaviate class: {self.collection_name}")
        else:
            print(f"✓ Weaviate class exists: {self.collection_name}")
    
    def add_documents(
        self, 
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            embeddings: Pre-computed embeddings
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        ids = []
        
        # Add documents in batch
        with self.client.batch as batch:
            for i, doc in enumerate(documents):
                # Generate UUID
                doc_id = str(uuid.uuid4())
                
                # Prepare data object
                data_object = {
                    "content": doc.content,
                    "metadata": str(doc.metadata)  # Store as string
                }
                
                # Add to batch
                if embeddings and i < len(embeddings):
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=self.collection_name,
                        uuid=doc_id,
                        vector=embeddings[i]
                    )
                else:
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=self.collection_name,
                        uuid=doc_id
                    )
                
                ids.append(doc_id)
        
        print(f"✓ Added {len(documents)} documents to Weaviate")
        
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
        """
        results_with_scores = self.similarity_search_with_score(query, k, filter)
        return [doc for doc, _ in results_with_scores]
    
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
        """
        # Note: This requires embedding function to be set
        # For simplicity, returning based on BM25 (keyword search)
        # In production, you'd use vector search with embeddings
        
        try:
            results = (
                self.client.query
                .get(self.collection_name, ["content", "metadata"])
                .with_limit(k)
                .with_additional(["certainty"])
                .with_near_text({"concepts": [query]})
                .do()
            )
            
            documents_with_scores = []
            
            if results.get("data", {}).get("Get", {}).get(self.collection_name):
                for item in results["data"]["Get"][self.collection_name]:
                    # Parse metadata
                    import ast
                    try:
                        metadata = ast.literal_eval(item["metadata"])
                    except:
                        metadata = {}
                    
                    doc = Document(
                        content=item["content"],
                        metadata=metadata
                    )
                    
                    # Get certainty score (0-1)
                    certainty = item.get("_additional", {}).get("certainty", 0.0)
                    
                    documents_with_scores.append((doc, certainty))
            
            return documents_with_scores
        
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.schema.delete_class(self.collection_name)
            print(f"✓ Deleted Weaviate class: {self.collection_name}")
        except Exception as e:
            print(f"✗ Error deleting class: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get object count
            result = (
                self.client.query
                .aggregate(self.collection_name)
                .with_meta_count()
                .do()
            )
            
            count = result.get("data", {}).get("Aggregate", {}).get(
                self.collection_name, [{}]
            )[0].get("meta", {}).get("count", 0)
            
            stats = {
                "collection_name": self.collection_name,
                "count": count,
                "url": self.url
            }
            
            return stats
        
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }