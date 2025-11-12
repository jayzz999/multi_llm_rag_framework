"""
Test Different Prompt Strategies
Compare zero-shot, few-shot, and chain-of-thought prompting
"""

import sys
sys.path.append('..')

from src.llms.ollama_llm import OllamaLLM
from src.prompt_engineering import PromptTemplate


def test_prompt_strategies():
    """Test different prompting strategies."""
    
    print("=" * 70)
    print("TESTING PROMPT STRATEGIES")
    print("=" * 70)
    print()
    
    # Initialize LLM
    print("Initializing LLM (Ollama - Llama 3.1)...")
    try:
        llm = OllamaLLM(model="llama3.1", temperature=0.7, max_tokens=300)
        print("✓ LLM ready\n")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print("Make sure Ollama is running and llama3.1 is installed")
        return
    
    # Sample context
    context = """Machine Learning is a subset of Artificial Intelligence that enables 
    systems to learn and improve from experience without being explicitly programmed. 
    It uses algorithms to analyze data, learn patterns, and make decisions. 
    Types include supervised learning (learning from labeled data), 
    unsupervised learning (finding patterns in unlabeled data), 
    and reinforcement learning (learning through trial and error)."""
    
    question = "What are the three types of machine learning?"
    
    print(f"Context: {context[:100]}...")
    print(f"Question: {question}")
    print("=" * 70)
    print()
    
    # Strategy 1: Zero-shot
    print("STRATEGY 1: ZERO-SHOT")
    print("-" * 70)
    
    prompt = PromptTemplate.zero_shot(context, question)
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        response = llm.generate(prompt)
        print(f"Answer: {response.content}")
        print(f"✓ Success\n")
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
    
    # Strategy 2: Few-shot
    print("STRATEGY 2: FEW-SHOT")
    print("-" * 70)
    
    examples = [
        {
            "question": "What is AI?",
            "answer": "AI (Artificial Intelligence) is the simulation of human intelligence by machines."
        },
        {
            "question": "What is deep learning?",
            "answer": "Deep learning is a subset of ML using neural networks with multiple layers."
        }
    ]
    
    prompt = PromptTemplate.few_shot(context, question, examples)
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Number of examples: {len(examples)}")
    
    try:
        response = llm.generate(prompt)
        print(f"Answer: {response.content}")
        print(f"✓ Success\n")
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
    
    # Strategy 3: Chain-of-thought
    print("STRATEGY 3: CHAIN-OF-THOUGHT")
    print("-" * 70)
    
    prompt = PromptTemplate.chain_of_thought(context, question)
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        response = llm.generate(prompt)
        print(f"Answer: {response.content}")
        print(f"✓ Success\n")
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
    
    # Strategy 4: With citations
    print("STRATEGY 4: WITH CITATIONS")
    print("-" * 70)
    
    numbered_context = f"""[Document 1]
{context}"""
    
    prompt = PromptTemplate.with_citations(numbered_context, question)
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        response = llm.generate(prompt)
        print(f"Answer: {response.content}")
        print(f"✓ Success\n")
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
    
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print("\nKey Observations:")
    print("- Zero-shot: Direct and concise")
    print("- Few-shot: Learns from examples, may follow format")
    print("- Chain-of-thought: Shows reasoning steps")
    print("- With citations: Includes source references")
    print("\n✓ Done!")


if __name__ == "__main__":
    test_prompt_strategies()