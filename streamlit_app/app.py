"""
Multi-LLM RAG Comparison - Streamlit App
Beautiful, modern UI for comparing different LLMs
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import create_rag_pipeline
from src.vector_stores.base import Document

# Set environment variables before importing
os.environ['USE_TF'] = '0'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Page config
st.set_page_config(
    page_title="Multi-LLM RAG Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }

    /* Main content */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    /* Headers */
    h1 {
        color: #1e293b;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h2 {
        color: #334155;
        font-weight: 700;
        margin-top: 2rem;
        border-bottom: 3px solid #6366f1;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #475569;
        font-weight: 600;
    }

    /* Cards */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        color: #6366f1;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Text input */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    /* Text area */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #6366f1;
    }

    /* Success messages */
    .success-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
    }

    /* LLM cards */
    .llm-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .llm-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border-color: #6366f1;
    }

    .llm-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .llm-meta {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .llm-response {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        margin-top: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'pipelines' not in st.session_state:
    st.session_state.pipelines = {}
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []

def check_api_keys():
    """Check which API keys are available."""
    available_llms = {}

    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        available_llms['OpenAI'] = [
            ('gpt-3.5-turbo', 'GPT-3.5 Turbo', '⚡ Fast & Cheap'),
            ('gpt-4', 'GPT-4', '🧠 Most Capable'),
        ]

    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        available_llms['Anthropic'] = [
            ('claude-3-haiku-20240307', 'Claude 3 Haiku', '⚡ Fastest Claude'),
        ]

    # Check if Ollama is running
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=1)
        if response.status_code == 200:
            available_llms['Ollama (Local)'] = [
                ('llama3.1', 'Llama 3.1', '🆓 Free Local'),
                ('mistral', 'Mistral', '🆓 Free Local'),
            ]
    except:
        pass

    return available_llms

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

def main():
    # Header
    st.title("🤖 Multi-LLM RAG Comparison")
    st.markdown("""
    <p style='text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>
    Compare different Large Language Models for Retrieval-Augmented Generation tasks
    </p>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Check available LLMs
        available_llms = check_api_keys()

        if not available_llms:
            st.error("⚠️ No API keys found! Please add your API keys to the .env file.")
            st.info("""
            Required environment variables:
            - OPENAI_API_KEY
            - ANTHROPIC_API_KEY
            """)
            return

        st.success(f"✅ {len(available_llms)} LLM provider(s) detected")

        # Show available models
        st.markdown("### 🤖 Available Models")
        total_models = sum(len(models) for models in available_llms.values())
        st.metric("Total Models", total_models)

        with st.expander("📋 View All Models"):
            for provider, models in available_llms.items():
                st.markdown(f"**{provider}**")
                for model_id, name, desc in models:
                    st.markdown(f"- {desc} {name}")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This app allows you to:
        - Add documents to a vector database
        - Query multiple LLMs simultaneously
        - Compare responses side-by-side
        - Analyze performance metrics
        """)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📚 Documents", "🔍 Compare LLMs", "📊 Results"])

    # TAB 1: Documents
    with tab1:
        st.header("📚 Document Management")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Add Documents to Knowledge Base")

            # Sample documents
            if st.button("🎯 Load Sample Documents", use_container_width=True):
                st.session_state.documents = create_sample_documents()
                st.success(f"✅ Loaded {len(st.session_state.documents)} sample documents!")
                st.balloons()

        with col2:
            st.metric("Documents Loaded", len(st.session_state.documents))

        # Custom document input
        st.markdown("### ✍️ Add Custom Document")
        doc_content = st.text_area("Document Content", height=150, placeholder="Enter your document text here...")
        doc_source = st.text_input("Source Name", placeholder="e.g., article.pdf")

        if st.button("➕ Add Document", use_container_width=True):
            if doc_content:
                new_doc = Document(
                    content=doc_content,
                    metadata={"source": doc_source or "custom", "topic": "Custom"}
                )
                st.session_state.documents.append(new_doc)
                st.success(f"✅ Document added! Total: {len(st.session_state.documents)}")
            else:
                st.warning("⚠️ Please enter document content")

        # Show loaded documents
        if st.session_state.documents:
            st.markdown("### 📋 Current Documents")
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"📄 Document {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                    st.markdown(f"**Topic:** {doc.metadata.get('topic', 'N/A')}")
                    st.markdown(f"**Content:**")
                    st.markdown(f'<div class="llm-response">{doc.content[:300]}...</div>', unsafe_allow_html=True)

    # TAB 2: Compare LLMs
    with tab2:
        st.header("🔍 Compare LLMs")

        if not st.session_state.documents:
            st.warning("⚠️ Please add documents in the 'Documents' tab first!")
            return

        # Select LLMs
        st.markdown("### Select LLMs to Compare")

        selected_llms = []
        cols = st.columns(min(len(available_llms), 4))

        for idx, (provider, models) in enumerate(available_llms.items()):
            with cols[idx % len(cols)]:
                st.markdown(f"**{provider}**")
                for model_id, name, desc in models:
                    if st.checkbox(f"{desc} {name}", key=model_id):
                        selected_llms.append({
                            'id': model_id,
                            'name': name,
                            'provider': provider.split()[0].lower(),
                            'display': f"{provider} - {name}"
                        })

        if not selected_llms:
            st.info("👆 Select at least one LLM above to start comparing")
            return

        st.success(f"✅ {len(selected_llms)} LLM(s) selected")

        # Query input
        st.markdown("### 💬 Enter Your Question")
        query = st.text_area(
            "Question",
            height=100,
            placeholder="e.g., What is RAG and why is it important?",
            label_visibility="collapsed"
        )

        # Compare button
        if st.button("🚀 Compare LLMs", type="primary", use_container_width=True):
            if not query:
                st.warning("⚠️ Please enter a question")
                return

            st.session_state.comparison_results = []

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, llm_config in enumerate(selected_llms):
                status_text.text(f"🔄 Querying {llm_config['display']}...")

                try:
                    # Create pipeline
                    collection_name = f"streamlit_{llm_config['provider']}_{llm_config['id'].replace('.', '_').replace('-', '_')}"

                    rag = create_rag_pipeline(
                        llm_type=llm_config['provider'],
                        llm_model=llm_config['id'],
                        vector_store_type="chroma",
                        embedding_type="openai",
                        collection_name=collection_name,
                        persist_directory=f"./streamlit_db/{collection_name}"
                    )

                    # Add documents if not already added
                    if collection_name not in st.session_state.pipelines:
                        rag.add_documents(st.session_state.documents, show_progress=False)
                        st.session_state.pipelines[collection_name] = True

                    # Query
                    start_time = time.time()
                    response = rag.query(query, top_k=3)
                    end_time = time.time()

                    st.session_state.comparison_results.append({
                        'llm': llm_config['display'],
                        'answer': response.answer,
                        'time': end_time - start_time,
                        'tokens': response.tokens_used,
                        'sources': len(response.source_documents)
                    })

                except Exception as e:
                    st.session_state.comparison_results.append({
                        'llm': llm_config['display'],
                        'answer': f"❌ Error: {str(e)}",
                        'time': 0,
                        'tokens': 0,
                        'sources': 0
                    })

                progress_bar.progress((idx + 1) / len(selected_llms))

            status_text.text("✅ Comparison complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            st.success("🎉 Comparison complete! Check the Results tab")

    # TAB 3: Results
    with tab3:
        st.header("📊 Comparison Results")

        if not st.session_state.comparison_results:
            st.info("👈 Run a comparison in the 'Compare LLMs' tab to see results here")
            return

        # Performance metrics
        st.markdown("### ⚡ Performance Overview")
        cols = st.columns(len(st.session_state.comparison_results))

        for idx, result in enumerate(st.session_state.comparison_results):
            with cols[idx]:
                st.metric(
                    result['llm'].split(' - ')[1] if ' - ' in result['llm'] else result['llm'],
                    f"{result['time']:.2f}s",
                    f"{result['tokens']} tokens"
                )

        # Detailed results
        st.markdown("### 📝 Detailed Responses")

        for result in st.session_state.comparison_results:
            st.markdown(f"""
            <div class="llm-card">
                <div class="llm-title">{result['llm']}</div>
                <div class="llm-meta">
                    ⏱️ {result['time']:.2f}s  |  🎯 {result['tokens']} tokens  |  📚 {result['sources']} sources
                </div>
                <div class="llm-response">
                    {result['answer']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Download results
        st.markdown("### 💾 Export Results")
        if st.button("📥 Download as JSON", use_container_width=True):
            import json
            json_str = json.dumps(st.session_state.comparison_results, indent=2)
            st.download_button(
                "Download JSON",
                json_str,
                "comparison_results.json",
                "application/json",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
