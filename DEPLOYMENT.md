# 🚀 Deployment Guide

This guide explains how to deploy your Multi-LLM RAG Comparison Framework to Streamlit Community Cloud.

## 📋 Prerequisites

1. GitHub account
2. Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
3. API keys for the LLMs you want to use

## 🎯 Deployment Strategy

### What Works in Cloud Deployment:
✅ **OpenAI** (GPT-3.5, GPT-4) - API-based, works perfectly
✅ **Anthropic Claude** - API-based, works perfectly
✅ **ChromaDB** - Lightweight vector DB, works great
✅ **OpenAI Embeddings** - Fast and reliable

### What Doesn't Work in Cloud Deployment:
❌ **Ollama** (Llama, Mistral) - Requires local installation, 4-7GB models
❌ **Sentence Transformers** - Large models, slow without GPU
⚠️ **Weaviate** - Requires external hosting

## 🌐 Streamlit Community Cloud Deployment

### Step 1: Prepare Your Repository

1. **Push to GitHub:**
```bash
git add .
git commit -m "Add Streamlit app for deployment"
git push origin main
```

2. **Create `.streamlit/secrets.toml` locally** (for testing):
```toml
# This file is NOT committed to git
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Set:
   - **Main file path**: `streamlit_app/app.py`
   - **Python version**: 3.11
   - **Requirements file**: Use custom requirements

5. Click "Advanced settings" and add your **Secrets**:
```toml
OPENAI_API_KEY = "sk-proj-your-actual-key-here"
ANTHROPIC_API_KEY = "sk-ant-your-actual-key-here"
```

6. Click "Deploy"!

### Step 3: Custom Requirements

Streamlit will look for `requirements.txt` by default. You have two options:

**Option A: Use `requirements-streamlit.txt`** (Recommended)
- Rename or specify this in deployment settings
- Lighter weight, faster deployment

**Option B: Modify `requirements.txt`**
- Remove heavy dependencies (torch, transformers, sentence-transformers)
- Keep only API-based dependencies

## 🏠 Local Development with All Features

To run locally with ALL LLMs including Ollama:

### 1. Install Full Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama
```bash
# macOS/Linux
curl https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.1
ollama pull mistral
```

### 3. Set Environment Variables
```bash
# Create .env file
cp .env.example .env

# Edit .env with your keys
```

### 4. Run Streamlit Locally
```bash
export USE_TF=0  # Avoid TensorFlow issues
streamlit run streamlit_app/app.py
```

## 🔐 Security Best Practices

1. **Never commit API keys** to git
   - Use `.env` files (already in `.gitignore`)
   - Use Streamlit secrets for deployment

2. **GitHub Secrets** are separate from Streamlit secrets
   - GitHub secrets are for CI/CD
   - Streamlit secrets are for the running app

3. **Test locally first** before deploying

## 📊 Deployment Comparison

| Feature | Streamlit Cloud | Local |
|---------|----------------|-------|
| **OpenAI** | ✅ Works | ✅ Works |
| **Anthropic** | ✅ Works | ✅ Works |
| **Ollama** | ❌ No | ✅ Works |
| **Cost** | Free | Free (API usage) |
| **Setup** | Easy | Medium |
| **Speed** | Fast (APIs) | Varies |

## 🎨 Customization

### Change App Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"  # Your brand color
backgroundColor = "#FFFFFF"
```

### Modify Models
Edit `streamlit_app/app.py`:
```python
# Add/remove models in check_api_keys() function
available_llms['openai'] = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
```

## 🐛 Troubleshooting

### "Module not found" Error
- Check that `requirements-streamlit.txt` includes all needed packages
- Don't include local-only packages (ollama, sentence-transformers)

### "API Key Not Found" Error
- Verify secrets are added in Streamlit Cloud settings
- Check exact key names match your code

### App is Slow
- Use OpenAI embeddings instead of sentence-transformers
- Reduce number of documents
- Lower top_k value

### ChromaDB Persistence Issues
- Streamlit Cloud may reset storage between sessions
- This is normal - app reinitializes on each run

## 📦 Recommended Cloud Configuration

For best performance on Streamlit Cloud:

```python
# Optimal settings
llm_type = "openai"  # or "anthropic"
embedding_type = "openai"
vector_store_type = "chroma"
top_k = 3  # Lower is faster
```

## 🚀 Next Steps

1. Deploy to Streamlit Cloud with OpenAI + Anthropic
2. Share your app URL!
3. Consider adding:
   - Custom document upload
   - More evaluation metrics
   - Cost tracking
   - Response quality ratings

## 📝 Example App URLs

After deployment, your app will be at:
```
https://[your-username]-multi-llm-rag.streamlit.app
```

## 💡 Tips

- Start with OpenAI only, add Anthropic later
- Test with sample documents first
- Monitor API usage costs
- Use caching for frequently asked questions
- Add analytics to track popular queries

---

**Need Help?** Check the [Streamlit Documentation](https://docs.streamlit.io) or open an issue!
