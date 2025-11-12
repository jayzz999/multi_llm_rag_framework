"""
RAG Pipeline
Orchestrates the entire Retrieval-Augmented Generation process
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.document_processor import DocumentProcessor
from src.embeddings.base import BaseEmbedding
from src.vector_stores.base import BaseVectorStore, Document
from src.llms.base import BaseLLM, LLMResponse


@dataclass
class RAGResponse:
    """
    Response from the RAG pipeline.
    """
    answer: str
    source_documents: List[Document]
    model: str
    tokens_used: Optional[int] = None
    retrieval_scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __repr__(self) -> str:
        preview = self.answer[:100] + "..." if len(self.answer) > 100 else self.answer
        return (f"RAGResponse(answer='{preview}', "
                f"sources={len(self.source_documents)})")


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline.
    
    Combines:
    - Document processing
    - Embedding generation
    - Vector storage
    - LLM generation
    
    Usage:
        # Initialize
        rag = RAGPipeline(
            llm=llm,
            vector_store=vector_store,
            embedding=embedding
        )
        
        # Add documents
        rag.add_documents(documents)
        
        # Query
        response = rag.query("What is RAG?")
        print(response.answer)
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        vector_store: BaseVectorStore,
        embedding: Optional[BaseEmbedding] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            llm: Language model for generation
            vector_store: Vector store for retrieval
            embedding: Embedding model (optional if vector store handles it)
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Example:
            from src.llms.ollama_llm import OllamaLLM
            from src.vector_stores.chroma_store import ChromaStore
            from src.embeddings.sentence_transformer import SentenceTransformerEmbedding
            
            llm = OllamaLLM(model="llama3.1")
            embedding = SentenceTransformerEmbedding()
            vector_store = ChromaStore(collection_name="docs")
            
            rag = RAGPipeline(llm=llm, vector_store=vector_store, embedding=embedding)
        """
        self.llm = llm
        self.vector_store = vector_store
        self.embedding = embedding
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        print("✓ RAG Pipeline initialized")
        print(f"  LLM: {llm.model}")
        print(f"  Vector Store: {vector_store.collection_name}")
        if embedding:
            print(f"  Embedding: {embedding.model_name}")
    
    def add_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            show_progress: Whether to show progress
            
        Returns:
            List of document IDs
            
        Example:
            from src.document_processor import process_documents
            
            docs = process_documents(["document.pdf"])
            ids = rag.add_documents(docs)
        """
        if not documents:
            print("⚠ No documents to add")
            return []
        
        if show_progress:
            print(f"Adding {len(documents)} documents to vector store...")
        
        # Generate embeddings if needed
        embeddings = None
        if self.embedding:
            if show_progress:
                print("  Generating embeddings...")
            
            texts = [doc.content for doc in documents]
            embeddings = self.embedding.embed_documents(texts)
            
            if show_progress:
                print(f"  ✓ Generated {len(embeddings)} embeddings")
        
        # Add to vector store
        ids = self.vector_store.add_documents(documents, embeddings)
        
        if show_progress:
            print(f"✓ Added {len(ids)} documents")
        
        return ids
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Query the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve (override default)
            return_sources: Whether to include source documents
            system_prompt: Optional system prompt for the LLM
            **kwargs: Additional parameters for LLM
            
        Returns:
            RAGResponse object
            
        Example:
            response = rag.query("What is machine learning?")
            print(response.answer)
            print(f"Sources: {len(response.source_documents)}")
        """
        k = top_k or self.top_k
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self._retrieve_documents(question, k)
        
        if not retrieved_docs:
            print("⚠ No relevant documents found")
            return RAGResponse(
                answer="I don't have enough information to answer that question.",
                source_documents=[],
                model=self.llm.model,
                metadata={"warning": "No documents retrieved"}
            )
        
        # Step 2: Construct context from retrieved documents
        context = self._construct_context(retrieved_docs)
        
        # Step 3: Generate answer using LLM
        answer = self._generate_answer(question, context, system_prompt, **kwargs)
        
        # Step 4: Create response
        response = RAGResponse(
            answer=answer.content,
            source_documents=retrieved_docs if return_sources else [],
            model=answer.model,
            tokens_used=answer.tokens_used,
            metadata={
                "num_sources": len(retrieved_docs),
                "context_length": len(context)
            }
        )
        
        return response
    
    def query_with_scores(
        self,
        question: str,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Query with similarity scores for retrieved documents.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            system_prompt: Optional system prompt
            **kwargs: Additional LLM parameters
            
        Returns:
            RAGResponse with retrieval scores
        """
        k = top_k or self.top_k

        # Generate query embedding if needed
        query_embedding = None
        if self.embedding:
            query_embedding = self.embedding.embed_query(question)

        # Retrieve with scores
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=question,
            k=k,
            query_embedding=query_embedding
        )
        
        # Filter by threshold
        docs_with_scores = [
            (doc, score) for doc, score in docs_with_scores
            if score >= self.similarity_threshold
        ]
        
        if not docs_with_scores:
            return RAGResponse(
                answer="I don't have enough information to answer that question.",
                source_documents=[],
                model=self.llm.model,
                retrieval_scores=[],
                metadata={"warning": "No documents above threshold"}
            )
        
        # Separate docs and scores
        retrieved_docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # Construct context and generate answer
        context = self._construct_context(retrieved_docs)
        answer = self._generate_answer(question, context, system_prompt, **kwargs)
        
        response = RAGResponse(
            answer=answer.content,
            source_documents=retrieved_docs,
            model=answer.model,
            tokens_used=answer.tokens_used,
            retrieval_scores=scores,
            metadata={
                "num_sources": len(retrieved_docs),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "context_length": len(context)
            }
        )
        
        return response
    
    def _retrieve_documents(
        self,
        query: str,
        k: int
    ) -> List[Document]:
        """
        Retrieve relevant documents from vector store.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        try:
            # Generate query embedding if needed
            query_embedding = None
            if self.embedding:
                query_embedding = self.embedding.embed_query(query)

            # Search vector store
            documents = self.vector_store.similarity_search(
                query=query,
                k=k,
                query_embedding=query_embedding
            )

            return documents

        except Exception as e:
            print(f"✗ Error retrieving documents: {str(e)}")
            return []
    
    def _construct_context(
        self,
        documents: List[Document]
    ) -> str:
        """
        Construct context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Add document with source info
            source_info = doc.metadata.get("source", "Unknown")
            
            context_part = f"[Document {i}]\n"
            context_part += f"Source: {source_info}\n"
            context_part += f"Content: {doc.content}\n"
            
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        return context
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            question: User question
            context: Retrieved context
            system_prompt: Optional system prompt
            **kwargs: Additional LLM parameters
            
        Returns:
            LLMResponse object
        """
        # Default system prompt for RAG
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. Answer questions based on the provided context. 
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite specific documents when using information from them."""
        
        # Construct the prompt
        prompt = f"""Context:
{context}

Question: {question}

Answer: Based on the context provided above, """
        
        # Generate response
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "llm": {
                "model": self.llm.model,
                "provider": self.llm.provider.value if self.llm.provider else "unknown"
            },
            "vector_store": self.vector_store.get_collection_stats(),
            "embedding": {
                "model": self.embedding.model_name if self.embedding else None,
                "dimension": self.embedding.dimension if self.embedding else None
            },
            "config": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold
            }
        }
        
        return stats
    
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        
        Example:
            rag.clear()
        """
        self.vector_store.delete_collection()
        print("✓ Cleared all documents")


# Convenience function
def create_rag_pipeline(
    llm_type: str = "ollama",
    llm_model: str = "llama3.1",
    vector_store_type: str = "chroma",
    embedding_type: str = "sentence-transformer",
    collection_name: str = "documents",
    **kwargs
) -> RAGPipeline:
    """
    Convenience function to create a RAG pipeline with sensible defaults.
    
    Args:
        llm_type: Type of LLM ("ollama", "openai", "claude")
        llm_model: Model name
        vector_store_type: Type of vector store ("chroma", "weaviate")
        embedding_type: Type of embedding ("sentence-transformer", "openai")
        collection_name: Name for the vector store collection
        **kwargs: Additional parameters
        
    Returns:
        Configured RAGPipeline
        
    Example:
        # Create with Ollama (FREE)
        rag = create_rag_pipeline(
            llm_type="ollama",
            llm_model="llama3.1",
            vector_store_type="chroma",
            embedding_type="sentence-transformer"
        )
        
        # Create with OpenAI
        rag = create_rag_pipeline(
            llm_type="openai",
            llm_model="gpt-4",
            vector_store_type="chroma",
            embedding_type="openai"
        )
    """
    # Import here to avoid circular imports
    from src.llms.ollama_llm import OllamaLLM
    from src.llms.openai_llm import OpenAILLM
    from src.llms.claude_llm import ClaudeLLM
    from src.llms.gemini_llm import GeminiLLM
    from src.vector_stores.chroma_store import ChromaStore
    from src.vector_stores.weaviate_store import WeaviateStore
    from src.embeddings.sentence_transformer import SentenceTransformerEmbedding
    from src.embeddings.openai_embeddings import OpenAIEmbedding

    # Initialize LLM
    if llm_type == "ollama":
        llm = OllamaLLM(model=llm_model)
    elif llm_type == "openai":
        llm = OpenAILLM(model=llm_model)
    elif llm_type in ["claude", "anthropic"]:
        llm = ClaudeLLM(model=llm_model)
    elif llm_type == "gemini":
        llm = GeminiLLM(model=llm_model)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    
    # Initialize embedding
    if embedding_type == "sentence-transformer":
        embedding = SentenceTransformerEmbedding()
    elif embedding_type == "openai":
        embedding = OpenAIEmbedding()
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    # Initialize vector store
    if vector_store_type == "chroma":
        vector_store = ChromaStore(
            collection_name=collection_name,
            persist_directory=kwargs.get("persist_directory", "./chroma_db")
        )
    elif vector_store_type == "weaviate":
        vector_store = WeaviateStore(
            collection_name=collection_name
        )
    else:
        raise ValueError(f"Unknown vector store type: {vector_store_type}")
    
    # Create pipeline
    pipeline = RAGPipeline(
        llm=llm,
        vector_store=vector_store,
        embedding=embedding,
        top_k=kwargs.get("top_k", 5)
    )
    
    return pipeline
