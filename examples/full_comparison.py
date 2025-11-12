"""
Full Multi-LLM Comparison
Compare all available LLMs side by side
"""

import sys
import os
sys.path.append('..')

from src.rag_pipeline import create_rag_pipeline
from src.vector_stores.base import Document
import time

# Set environment variables
os.environ['USE_TF'] = '0'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def create_sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            content="""Retrieval-Augmented Generation (RAG) is a technique that combines
            information retrieval with language model generation. It retrieves relevant
            documents from a knowledge base and uses them as context for generating responses.
            This approach significantly reduces hallucinations and improves factual accuracy.""",
            metadata={"source": "rag_intro.txt", "topic": "RAG"}
        ),
        Document(
            content="""Vector databases store embeddings of text documents and enable
            fast similarity search. Popular vector databases include ChromaDB, Weaviate,
            Pinecone, and FAISS. They are essential for RAG systems as they allow semantic
            search rather than just keyword matching.""",
            metadata={"source": "vector_db.txt", "topic": "Vector Databases"}
        ),
        Document(
            content="""Large Language Models (LLMs) like GPT-4, Claude, and Llama are
            neural networks trained on vast amounts of text. They can generate human-like
            text but may hallucinate without proper grounding in factual information.
            RAG helps mitigate this by providing relevant context.""",
            metadata={"source": "llm_basics.txt", "topic": "LLMs"}
        ),
        Document(
            content="""Prompt engineering is the practice of designing effective prompts
            for LLMs. Techniques include zero-shot, few-shot, chain-of-thought, and
            ReAct prompting. Good prompts significantly improve model performance and
            reduce errors.""",
            metadata={"source": "prompting.txt", "topic": "Prompt Engineering"}
        ),
        Document(
            content="""Embeddings are vector representations of text that capture semantic
            meaning. Similar texts have similar embeddings. They enable semantic search
            rather than just keyword matching. Common embedding models include OpenAI's
            text-embedding-3-small and sentence transformers.""",
            metadata={"source": "embeddings.txt", "topic": "Embeddings"}
        )
    ]


def run_comparison():
    """Run full multi-LLM comparison."""

    print("=" * 80)
    print("FULL MULTI-LLM RAG COMPARISON")
    print("=" * 80)
    print()

    # Define all LLMs to test
    llm_configs = [
        # OpenAI Models
        {"type": "openai", "model": "gpt-3.5-turbo", "name": "OpenAI GPT-3.5 Turbo"},
        {"type": "openai", "model": "gpt-4", "name": "OpenAI GPT-4"},
        {"type": "openai", "model": "gpt-4-turbo-preview", "name": "OpenAI GPT-4 Turbo"},

        # Anthropic Models
        {"type": "anthropic", "model": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
        {"type": "anthropic", "model": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
        {"type": "anthropic", "model": "claude-3-opus-20240229", "name": "Claude 3 Opus"},

        # Google Gemini Models
        {"type": "gemini", "model": "gemini-pro", "name": "Google Gemini Pro"},

        # Ollama Models (FREE - only if Ollama is running)
        {"type": "ollama", "model": "llama3.1", "name": "Llama 3.1 (Ollama)"},
        {"type": "ollama", "model": "mistral", "name": "Mistral (Ollama)"},
    ]

    # Questions to ask
    questions = [
        "What is RAG and why is it important?",
        "Which vector databases are commonly used for RAG?",
        "How do embeddings enable semantic search?",
    ]

    # Create sample documents
    print("📄 Creating sample documents...")
    documents = create_sample_documents()
    print(f"✓ Created {len(documents)} documents\n")

    results = []

    # Test each LLM
    for idx, config in enumerate(llm_configs, 1):
        print(f"\n[{idx}/{len(llm_configs)}] Testing {config['name']}...")
        print("-" * 80)

        try:
            # Check if API key is available
            if config['type'] == 'openai' and not os.getenv('OPENAI_API_KEY'):
                print(f"⚠️  OpenAI API key not found, skipping...")
                continue
            if config['type'] == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
                print(f"⚠️  Anthropic API key not found, skipping...")
                continue
            if config['type'] == 'gemini' and not os.getenv('GEMINI_API_KEY'):
                print(f"⚠️  Gemini API key not found, skipping...")
                continue
            if config['type'] == 'ollama':
                # Check if Ollama is running
                try:
                    import requests
                    response = requests.get('http://localhost:11434/api/tags', timeout=2)
                    if response.status_code != 200:
                        print(f"⚠️  Ollama not running, skipping...")
                        continue
                except:
                    print(f"⚠️  Ollama not available, skipping...")
                    continue

            # Create pipeline
            collection_name = f"compare_{config['type']}_{config['model'].replace('.', '_').replace('-', '_')}"

            rag = create_rag_pipeline(
                llm_type=config['type'],
                llm_model=config['model'],
                vector_store_type="chroma",
                embedding_type="openai",
                collection_name=collection_name,
                persist_directory=f"./comparison_db/{collection_name}"
            )

            print(f"✓ Pipeline initialized")

            # Add documents
            print("Adding documents...")
            rag.add_documents(documents, show_progress=False)
            print(f"✓ Documents added\n")

            # Test each question
            llm_results = {
                "name": config['name'],
                "model": config['model'],
                "type": config['type'],
                "questions": []
            }

            for q_idx, question in enumerate(questions, 1):
                print(f"  Question {q_idx}: {question}")

                start_time = time.time()
                response = rag.query(question, top_k=3)
                end_time = time.time()

                llm_results["questions"].append({
                    "question": question,
                    "answer": response.answer,
                    "time": end_time - start_time,
                    "tokens": response.tokens_used,
                    "sources": len(response.source_documents)
                })

                print(f"    ✓ Answered in {end_time - start_time:.2f}s ({response.tokens_used} tokens)")

            results.append(llm_results)

            # Cleanup
            rag.clear()
            print(f"\n✅ {config['name']} completed!\n")

        except Exception as e:
            print(f"❌ Error with {config['name']}: {str(e)}\n")
            continue

    # Display results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    if not results:
        print("❌ No LLMs were successfully tested")
        return

    # Show results for each question
    for q_idx, question in enumerate(questions, 1):
        print(f"\n📊 Question {q_idx}: {question}")
        print("-" * 80)

        for result in results:
            if q_idx - 1 < len(result["questions"]):
                q_result = result["questions"][q_idx - 1]
                print(f"\n🤖 {result['name']}:")
                print(f"   Time: {q_result['time']:.2f}s | Tokens: {q_result['tokens']} | Sources: {q_result['sources']}")
                print(f"   Answer: {q_result['answer'][:200]}...")

    # Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    for result in results:
        avg_time = sum(q['time'] for q in result['questions']) / len(result['questions'])
        total_tokens = sum(q['tokens'] for q in result['questions'])

        print(f"\n{result['name']}:")
        print(f"  Average Response Time: {avg_time:.2f}s")
        print(f"  Total Tokens Used: {total_tokens}")
        print(f"  Questions Answered: {len(result['questions'])}")

    print("\n" + "=" * 80)
    print(f"✅ Comparison complete! Tested {len(results)} LLMs on {len(questions)} questions")
    print("=" * 80)


if __name__ == "__main__":
    # Ensure API keys are set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  Warning: ANTHROPIC_API_KEY not found in environment")

    run_comparison()
