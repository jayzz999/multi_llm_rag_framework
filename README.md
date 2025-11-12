# 🚀 Multi-LLM RAG Comparison Framework

A comprehensive framework for comparing Retrieval-Augmented Generation (RAG) implementations across multiple Large Language Models, vector databases, and embedding models.

## 📋 Overview

This project compares:
- **LLMs**: OpenAI GPT-4, Anthropic Claude, Llama 3.1, Mistral
- **Vector Databases**: ChromaDB, Weaviate, FAISS
- **Embeddings**: OpenAI, Sentence Transformers
- **Prompt Strategies**: Zero-shot, Few-shot, Chain-of-thought

## 🎯 Purpose

Built specifically to demonstrate proficiency in:
- LLM integration and orchestration
- RAG pipeline implementation
- Vector database operations
- Prompt engineering techniques
- Comparative evaluation and benchmarking

## 🛠️ Tech Stack

- Python 3.10+
- LangChain
- ChromaDB, Weaviate, FAISS
- OpenAI, Anthropic, Ollama
- Streamlit
- Sentence Transformers

## 🚀 Quick Start

### 1. Clone & Setup
```bash
# Create project directory
mkdir multi_llm_rag_framework
cd multi_llm_rag_framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Ollama (for local LLMs)
```bash
# Mac/Linux
curl https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai

# Pull models
ollama pull llama3.1
ollama pull mistral
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# (Optional: Start with Ollama only - no API keys needed!)
```

### 4. Run Examples
```bash
# Coming soon - we'll add examples as we build!
```

## 📁 Project Structure
```
multi_llm_rag_framework/
├── src/                    # Source code
├── config/                 # Configuration files
├── data/                   # Data storage
├── examples/               # Usage examples
├── tests/                  # Unit tests
└── streamlit_app/          # Web interface
```

## 🎓 Features

- [ ] Document processing (PDF, TXT, MD)
- [ ] Multiple vector database support
- [ ] Multi-LLM integration
- [ ] RAG pipeline implementation
- [ ] Prompt engineering templates
- [ ] Comprehensive evaluation metrics
- [ ] Interactive Streamlit dashboard
- [ ] Benchmarking tools

## 📊 Status

🚧 **In Development** - Building step by step

## 👤 Author

**Jayanth Muthina**
- Email: jayanthmuthina852@gmail.com
- LinkedIn: [linkedin.com/in/jayanth-muthina](https://linkedin.com/in/jayanth-muthina)
- GitHub: [github.com/jayzz999](https://github.com/jayzz999)

## 📄 License

MIT License

---

⭐ Star this project if you find it useful!
```

---

## ✅ PHASE 1 COMPLETE! 🎉

You now have all 4 setup files:
1. ✅ `.gitignore`
2. ✅ `requirements.txt`
3. ✅ `.env.example`
4. ✅ `README.md`

---

## 📦 YOUR PROJECT SO FAR:
```
multi_llm_rag_framework/
├── .gitignore
├── requirements.txt
├── .env.example
└── README.md