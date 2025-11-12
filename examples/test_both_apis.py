"""
Test Both OpenAI and Anthropic APIs
Quick verification that both API keys work
"""

import sys
import os
sys.path.append('..')

from src.rag_pipeline import create_rag_pipeline
from src.vector_stores.base import Document

# Set environment variable
os.environ['USE_TF'] = '0'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def test_api(llm_type, llm_model):
    """Test a specific LLM API."""
    print(f"\n{'='*60}")
    print(f"Testing {llm_type.upper()} - {llm_model}")
    print(f"{'='*60}")

    try:
        # Create RAG pipeline
        print(f"Initializing {llm_model}...")
        rag = create_rag_pipeline(
            llm_type=llm_type,
            llm_model=llm_model,
            vector_store_type="chroma",
            embedding_type="openai",
            collection_name=f"test_{llm_type}_{llm_model.replace('.', '_').replace('-', '_')}",
            persist_directory=f"./test_db_{llm_type}"
        )
        print("✓ Pipeline initialized\n")

        # Add test document
        test_doc = Document(
            content="RAG stands for Retrieval-Augmented Generation. It combines retrieval with generation.",
            metadata={"source": "test.txt"}
        )

        print("Adding test document...")
        rag.add_documents([test_doc], show_progress=False)
        print("✓ Document added\n")

        # Query
        question = "What does RAG stand for?"
        print(f"Question: {question}")
        print("-" * 60)

        response = rag.query(question, top_k=1)

        print(f"Answer: {response.answer}")
        print(f"\nModel: {response.model}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Sources: {len(response.source_documents)}")

        # Cleanup
        rag.clear()

        print(f"\n✅ SUCCESS: {llm_type.upper()} - {llm_model} is working!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {llm_type.upper()} - {llm_model}")
        print(f"Error: {str(e)}")
        return False


def main():
    """Test all available LLMs."""
    print("\n" + "="*60)
    print("MULTI-LLM API TEST")
    print("="*60)

    results = []

    # Test OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("\n[1/2] Testing OpenAI...")
        success = test_api("openai", "gpt-3.5-turbo")
        results.append(("OpenAI GPT-3.5", success))
    else:
        print("\n⚠️ OpenAI API key not found")
        results.append(("OpenAI GPT-3.5", False))

    # Test Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n[2/2] Testing Anthropic...")
        success = test_api("anthropic", "claude-3-haiku-20240307")
        results.append(("Anthropic Claude", success))
    else:
        print("\n⚠️ Anthropic API key not found")
        results.append(("Anthropic Claude", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All APIs are working! Ready to deploy!")
    elif passed > 0:
        print(f"\n⚠️ Some APIs working. You can use {passed} LLM(s) for comparison.")
    else:
        print("\n❌ No APIs working. Check your API keys in .env file")

    print("="*60)


if __name__ == "__main__":
    main()
