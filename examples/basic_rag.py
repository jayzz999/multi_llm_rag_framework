"""
Basic RAG Example
Simple demonstration of the RAG pipeline
"""

import sys
sys.path.append('..')

from src.rag_pipeline import create_rag_pipeline
from src.document_processor import DocumentProcessor
from src.vector_stores.base import Document


def main():
    """Run a basic RAG example."""
    
    print("=" * 60)
    print("BASIC RAG EXAMPLE")
    print("=" * 60)
    print()
    
    # Step 1: Create sample documents
    print("Step 1: Creating sample documents...")
    sample_docs = [
        Document(
            content="""Retrieval-Augmented Generation (RAG) is a technique that combines 
            information retrieval with language model generation. It retrieves relevant 
            documents from a knowledge base and uses them as context for generating responses.""",
            metadata={"source": "rag_intro.txt", "topic": "RAG"}
        ),
        Document(
            content="""Vector databases store embeddings of text documents and enable 
            fast similarity search. Popular vector databases include ChromaDB, Weaviate, 
            Pinecone, and FAISS. They are essential for RAG systems.""",
            metadata={"source": "vector_db.txt", "topic": "Vector Databases"}
        ),
        Document(
            content="""Large Language Models (LLMs) like GPT-4, Claude, and Llama are 
            neural networks trained on vast amounts of text. They can generate human-like 
            text but may hallucinate without proper grounding in factual information.""",
            metadata={"source": "llm_basics.txt", "topic": "LLMs"}
        ),
        Document(
            content="""Prompt engineering is the practice of designing effective prompts 
            for LLMs. Techniques include zero-shot, few-shot, chain-of-thought, and 
            ReAct prompting. Good prompts significantly improve model performance.""",
            metadata={"source": "prompting.txt", "topic": "Prompt Engineering"}
        ),
        Document(
            content="""Embeddings are vector representations of text that capture semantic 
            meaning. Similar texts have similar embeddings. They enable semantic search 
            rather than just keyword matching.""",
            metadata={"source": "embeddings.txt", "topic": "Embeddings"}
        )
    ]
    print(f"✓ Created {len(sample_docs)} sample documents")
    print()
    
    # Step 2: Initialize RAG pipeline with FREE components
    print("Step 2: Initializing RAG pipeline (using FREE components)...")
    print("  - LLM: Ollama (Llama 3.1) - FREE")
    print("  - Embedding: Sentence Transformer - FREE")
    print("  - Vector Store: ChromaDB - FREE")
    print()
    
    try:
        rag = create_rag_pipeline(
            llm_type="ollama",
            llm_model="llama3.1",
            vector_store_type="chroma",
            embedding_type="sentence-transformer",
            collection_name="basic_example",
            persist_directory="./example_db"
        )
        print("✓ RAG pipeline initialized")
        print()
    
    except Exception as e:
        print(f"✗ Error initializing pipeline: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is installed: https://ollama.ai")
        print("2. Pull the model: ollama pull llama3.1")
        print("3. Make sure Ollama is running")
        return
    
    # Step 3: Add documents to the pipeline
    print("Step 3: Adding documents to vector store...")
    try:
        rag.add_documents(sample_docs)
        print()
    except Exception as e:
        print(f"✗ Error adding documents: {str(e)}")
        return
    
    # Step 4: Query the RAG system
    print("Step 4: Querying the RAG system...")
    print("-" * 60)
    
    questions = [
        "What is RAG?",
        "Which vector databases are mentioned?",
        "What are some prompt engineering techniques?",
        "How do embeddings work?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 60)
        
        try:
            response = rag.query(question, top_k=3)
            
            print(f"Answer: {response.answer}")
            print(f"\nSources used: {response.metadata['num_sources']}")
            print(f"Model: {response.model}")
            if response.tokens_used:
                print(f"Tokens: {response.tokens_used}")
            
            print("\nSource documents:")
            for j, doc in enumerate(response.source_documents, 1):
                source = doc.metadata.get("source", "Unknown")
                print(f"  {j}. {source}")
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        
        print("-" * 60)
    
    # Step 5: Show pipeline stats
    print("\n" + "=" * 60)
    print("PIPELINE STATISTICS")
    print("=" * 60)
    stats = rag.get_stats()
    print(f"LLM: {stats['llm']['model']} ({stats['llm']['provider']})")
    print(f"Vector Store: {stats['vector_store']['collection_name']}")
    print(f"Documents: {stats['vector_store']['count']}")
    print(f"Embedding Model: {stats['embedding']['model']}")
    print(f"Embedding Dimension: {stats['embedding']['dimension']}")
    print("=" * 60)
    
    # Cleanup
    print("\nCleaning up...")
    rag.clear()
    print("✓ Done!")


if __name__ == "__main__":
    main()