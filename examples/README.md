# Examples

Simple examples demonstrating the Multi-LLM RAG Framework.

## Prerequisites

1. **Install Ollama**: https://ollama.ai
2. **Pull models**:
```bash
   ollama pull llama3.1
   ollama pull mistral
```
3. **Install dependencies**: Already done via `requirements.txt`

## Running Examples

### 1. Basic RAG Example

Demonstrates the complete RAG pipeline with sample documents.
```bash
python basic_rag.py
```

**What it does:**
- Creates sample documents
- Initializes RAG pipeline (100% FREE components)
- Adds documents to vector store
- Queries the system
- Shows results and statistics

### 2. Compare LLMs

Tests multiple LLMs on the same question.
```bash
python compare_llms.py
```

**What it does:**
- Compares Llama 3.1 vs Mistral
- Measures response time
- Shows speed comparison

### 3. Test Prompt Strategies

Demonstrates different prompting techniques.
```bash
python test_prompt_strategies.py
```

**What it does:**
- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought
- With citations

## Troubleshooting

**Error: "ollama package not installed"**
```bash
pip install ollama
```

**Error: "Model not found"**
```bash
ollama pull llama3.1
```

**Error: "Connection refused"**
- Make sure Ollama is running
- Check: `ollama list`

## Next Steps

After running examples:
1. Try with your own documents
2. Experiment with different models
3. Test various prompt strategies
4. Build your own use cases!
