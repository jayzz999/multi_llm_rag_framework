"""
OpenAI RAG Example - Optimized for Performance
Uses OpenAI for both LLM and embeddings (fast, no local model loading)
"""

import sys
sys.path.append('..')

from src.rag_pipeline import create_rag_pipeline
from src.vector_stores.base import Document


def main():
    """Run an optimized RAG example using OpenAI."""

    print("=" * 60)
    print("OPENAI RAG EXAMPLE - OPTIMIZED")
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

    # Step 2: Initialize RAG pipeline with OpenAI (FAST & LIGHTWEIGHT)
    print("Step 2: Initializing RAG pipeline (using OpenAI)...")
    print("  - LLM: OpenAI GPT-3.5-turbo (Fast & Cheap)")
    print("  - Embedding: OpenAI text-embedding-3-small (No local model loading!)")
    print("  - Vector Store: ChromaDB (Local)")
    print()

    try:
        rag = create_rag_pipeline(
            llm_type="openai",
            llm_model="gpt-3.5-turbo",  # Fast and affordable
            vector_store_type="chroma",
            embedding_type="openai",  # Cloud-based, no heavy models
            collection_name="openai_example",
            persist_directory="./openai_db"
        )
        print("✓ RAG pipeline initialized (no lag!)")
        print()

    except Exception as e:
        print(f"✗ Error initializing pipeline: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure OPENAI_API_KEY is set in .env file")
        print("2. Check your OpenAI API key is valid")
        print("3. Ensure you have internet connection")
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

    # Performance Notes
    print("\nPerformance Benefits:")
    print("✓ No heavy models loaded into memory")
    print("✓ Fast startup time")
    print("✓ Minimal CPU/RAM usage")
    print("✓ Cloud-based embeddings (no GPU needed)")

    # Cleanup
    print("\nCleaning up...")
    rag.clear()
    print("✓ Done!")


if __name__ == "__main__":
    main()
