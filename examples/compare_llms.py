"""
Compare Multiple LLMs
Test the same query across different LLMs
"""

import sys
sys.path.append('..')

import time
from src.rag_pipeline import RAGPipeline
from src.llms.ollama_llm import OllamaLLM
from src.vector_stores.chroma_store import ChromaStore
from src.embeddings.sentence_transformer import SentenceTransformerEmbedding
from src.vector_stores.base import Document


def compare_llms():
    """Compare different LLMs on the same question."""
    
    print("=" * 70)
    print("COMPARING MULTIPLE LLMs")
    print("=" * 70)
    print()
    
    # Sample document
    sample_doc = Document(
        content="""Artificial Intelligence (AI) is transforming multiple industries. 
        Machine learning enables computers to learn from data without explicit programming. 
        Deep learning, a subset of ML, uses neural networks with multiple layers. 
        Natural Language Processing (NLP) allows machines to understand and generate human language. 
        Computer vision enables machines to interpret visual information. 
        AI applications include autonomous vehicles, medical diagnosis, recommendation systems, 
        and voice assistants.""",
        metadata={"source": "ai_overview.txt"}
    )
    
    # Setup embedding and vector store (shared across LLMs)
    print("Setting up shared components...")
    embedding = SentenceTransformerEmbedding()
    vector_store = ChromaStore(
        collection_name="llm_comparison",
        persist_directory="./comparison_db"
    )
    
    # Add document
    print("Adding document to vector store...")
    embeddings = embedding.embed_documents([sample_doc.content])
    vector_store.add_documents([sample_doc], embeddings)
    print("✓ Document added\n")
    
    # Define LLMs to compare
    llm_configs = [
        {
            "name": "Llama 3.1 (8B)",
            "model": "llama3.1",
            "type": "ollama"
        },
        {
            "name": "Mistral (7B)",
            "model": "mistral",
            "type": "ollama"
        }
    ]
    
    # Test question
    question = "What are the main applications of AI mentioned in the context?"
    
    print(f"Question: {question}")
    print("=" * 70)
    print()
    
    results = []
    
    # Test each LLM
    for config in llm_configs:
        print(f"Testing: {config['name']}")
        print("-" * 70)
        
        try:
            # Initialize LLM
            if config['type'] == 'ollama':
                llm = OllamaLLM(model=config['model'])
            else:
                print(f"  ⚠ Skipping (not configured)")
                continue
            
            # Create RAG pipeline
            rag = RAGPipeline(
                llm=llm,
                vector_store=vector_store,
                embedding=embedding,
                top_k=1
            )
            
            # Query
            start_time = time.time()
            response = rag.query(question)
            elapsed_time = time.time() - start_time
            
            # Store results
            results.append({
                "name": config['name'],
                "model": config['model'],
                "answer": response.answer,
                "time": elapsed_time,
                "tokens": response.tokens_used
            })
            
            print(f"  Answer: {response.answer[:200]}...")
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  ✓ Success")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            print(f"  Make sure '{config['model']}' is installed:")
            print(f"    ollama pull {config['model']}")
        
        print()
    
    # Summary comparison
    if results:
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        for result in results:
            print(f"\n{result['name']}:")
            print(f"  Model: {result['model']}")
            print(f"  Response Time: {result['time']:.2f}s")
            if result['tokens']:
                tokens_per_sec = result['tokens'] / result['time']
                print(f"  Tokens: {result['tokens']} ({tokens_per_sec:.1f} tok/s)")
            print(f"  Answer: {result['answer'][:150]}...")
        
        print("\n" + "=" * 70)
        
        # Find fastest
        fastest = min(results, key=lambda x: x['time'])
        print(f"Fastest: {fastest['name']} ({fastest['time']:.2f}s)")
    
    # Cleanup
    print("\nCleaning up...")
    vector_store.delete_collection()
    print("✓ Done!")


if __name__ == "__main__":
    compare_llms()