# 🚀 Quick Start Guide

## ✅ What You Have Configured

Your `.env` file is set up with:
- ✅ OpenAI API Key (GPT-3.5, GPT-4)
- ✅ Anthropic API Key (Claude)
- ✅ ChromaDB (local vector database)

## 🎯 Three Ways to Use Your Project

### 1️⃣ Run Streamlit App (Recommended for Demos)

```bash
# Set environment variable to avoid TensorFlow issues
export USE_TF=0

# Run the app
streamlit run streamlit_app/app.py
```

**What it does:**
- Beautiful web interface
- Compare OpenAI vs Anthropic side-by-side
- Interactive document management
- Performance metrics and visualizations
- **No lag** - uses only API-based models

**Perfect for:**
- Demonstrating to recruiters/employers
- Portfolio projects
- Interactive testing

---

### 2️⃣ Run Python Examples (Good for Testing)

```bash
# Test with OpenAI only (fast, no lag)
export USE_TF=0
python examples/openai_rag.py

# Compare multiple LLMs
python examples/compare_llms.py
```

**What it does:**
- Command-line interface
- Quick testing
- See raw outputs

---

### 3️⃣ Deploy to Cloud (Share with Others)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Quick steps:**
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add API keys as secrets
5. Deploy!

**Result:**
- Public URL you can share
- Works on any device
- No installation needed for users

## 💡 About Ollama (Local LLMs)

### ❌ Why We're NOT Using Ollama for Cloud Deployment:

1. **Size**: Each model is 4-7GB
2. **Performance**: Slow without GPU
3. **Cloud Limitations**: Can't install on Streamlit Cloud
4. **Lag Issues**: This is what caused your system to lag!

### ✅ What We're Using Instead:

**OpenAI + Anthropic (Cloud APIs):**
- Fast responses
- No local model loading
- Works everywhere (local + cloud)
- No system lag
- Still provides true multi-LLM comparison!

### 🏠 If You Want Ollama Locally (Optional):

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.1
ollama pull mistral

# Modify Streamlit app to include Ollama
# (It will auto-detect if Ollama is running)
```

## 🎨 Current Capabilities

Your app can compare:

| LLM | Speed | Cost | Cloud Deploy |
|-----|-------|------|--------------|
| GPT-3.5 Turbo | ⚡ Fast | 💰 Cheap | ✅ Yes |
| GPT-4 | 🐢 Slower | 💰💰 Expensive | ✅ Yes |
| Claude Haiku | ⚡ Fast | 💰 Cheap | ✅ Yes |
| Claude Sonnet | ⚡ Fast | 💰💰 Medium | ✅ Yes |

**Embeddings:**
- OpenAI text-embedding-3-small (fast, 1536 dim)

**Vector Database:**
- ChromaDB (lightweight, persistent)

## 📊 What Makes This a Great Portfolio Project

1. **Multi-LLM Integration** ✅
   - OpenAI GPT models
   - Anthropic Claude models
   - Easy to add more

2. **RAG Implementation** ✅
   - Document processing
   - Vector embeddings
   - Semantic search
   - Context-aware generation

3. **Production-Ready** ✅
   - Cloud deployable
   - No performance issues
   - Professional UI
   - Scalable architecture

4. **Best Practices** ✅
   - Environment variables
   - Type hints
   - Error handling
   - Clean code structure

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'tf_keras'"
```bash
export USE_TF=0
```
Add this before running any Python command.

### Streamlit Won't Start
```bash
# Kill existing processes
pkill -f streamlit

# Try again
streamlit run streamlit_app/app.py
```

### API Key Errors
Check your `.env` file:
```bash
cat .env | grep API_KEY
```

Should show your keys (not the example placeholders).

## 📝 Next Steps

1. **Test the Streamlit app:**
   ```bash
   export USE_TF=0
   streamlit run streamlit_app/app.py
   ```

2. **Try comparing LLMs:**
   - Load sample documents
   - Select both OpenAI and Anthropic models
   - Ask a question
   - See side-by-side comparison!

3. **Deploy to cloud:**
   - Follow [DEPLOYMENT.md](DEPLOYMENT.md)
   - Share your app URL on LinkedIn/GitHub!

4. **Customize:**
   - Add your own documents
   - Try different models
   - Adjust parameters
   - Add evaluation metrics

## 🎯 Recommended for Job Applications

When showing this project:

1. **Live Demo**: Deploy to Streamlit Cloud, share the link
2. **GitHub README**: Add screenshots of the comparison UI
3. **Technical Discussion**: Explain RAG architecture, embedding strategies
4. **Results**: Show performance comparisons, response quality

---

**Questions?** Check the [README.md](README.md) or [DEPLOYMENT.md](DEPLOYMENT.md)!
