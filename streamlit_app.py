import os
import streamlit as st
from src.config import PINECONE_INDEX_NAME, OPENAI_API_KEY, PINECONE_API_KEY
from src.graph import build_rag_graph

# Page configuration
st.set_page_config(
    page_title="Agentic AI RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Ultra-Modern CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Hero Header */
    .hero-container {
        padding: 1.5rem 1.8rem;
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        backdrop-filter: blur(16px);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.5rem;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Sidebar Custom Styling */
    .sidebar-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 1rem;
    }
    .sidebar-card-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.88rem;
    }
    .stat-row:last-child {
        border-bottom: none;
    }
    .stat-label {
        color: #94a3b8;
    }
    .stat-val {
        color: #f1f5f9;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Inspector Score Card */
    .inspector-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1rem;
    }
    .score-card-v2 {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
    }
    .score-high {
        border-left: 4px solid #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
    }
    .score-medium {
        border-left: 4px solid #f59e0b;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
    }
    .score-refusal {
        border-left: 4px solid #ef4444;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
    }
    .score-number {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        line-height: 1;
        margin: 6px 0;
    }
    .score-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Progress bar */
    .progress-bar-bg {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 9999px;
        overflow: hidden;
        margin: 10px 0;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Chunk cards */
    .chunk-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #cbd5e1;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .chunk-box:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }
    .chunk-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 6px;
    }
    .chunk-content {
        white-space: pre-wrap;
        font-family: inherit;
        font-size: 0.85rem;
        color: #e2e8f0;
        max-height: 220px;
        overflow-y: auto;
        padding-right: 6px;
    }

    /* Welcome Hero Card for Empty State */
    .welcome-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px dashed rgba(99, 102, 241, 0.3);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 20px 0;
    }
    .welcome-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 8px;
    }
    .welcome-desc {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 520px;
        margin: 0 auto 16px auto;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 16px;
    }
    .feature-item {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 12px;
        text-align: left;
    }
    .feature-item-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #818cf8;
        margin-bottom: 4px;
    }
    .feature-item-desc {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top Hero Section
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">⚡ Agentic AI RAG Studio</div>
        <div class="hero-subtitle">
            Enterprise Retrieval-Augmented Generation system grounded strictly on the <strong>Agentic AI eBook</strong> (60 pages). 
            Powered by <strong>LangGraph</strong> stateful execution, <strong>Pinecone</strong> vector indexing, and anti-hallucination guardrails.
        </div>
        <div class="hero-badges">
            <span class="badge-pill badge-purple">LangGraph StateGraph</span>
            <span class="badge-pill badge-blue">Pinecone Vector DB (1536-D)</span>
            <span class="badge-pill badge-emerald">Strict Document Grounding</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Design
with st.sidebar:
    st.markdown('<div class="sidebar-card-title">⚙️ System & Engine Status</div>', unsafe_allow_html=True)
    
    is_live = bool(OPENAI_API_KEY) and bool(PINECONE_API_KEY) and not os.getenv("MOCK_MODE", "").lower() in ("true", "1")

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="stat-row">
                <span class="stat-label">Vector Index:</span>
                <span class="stat-val">{PINECONE_INDEX_NAME}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Source Document:</span>
                <span class="stat-val">Ebook-Agentic-AI.pdf</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Document Chunks:</span>
                <span class="stat-val">119 Chunks</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Pinecone Status:</span>
                <span class="stat-val" style="color: {'#34d399' if PINECONE_API_KEY else '#f87171'}">{'● Connected' if PINECONE_API_KEY else '○ Not Set'}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Execution Mode:</span>
                <span class="stat-val" style="color: #818cf8">{'🟢 Live Cloud' if is_live else '🟡 Local Studio'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-card-title">💡 Benchmark Verification Queries</div>', unsafe_allow_html=True)
    st.caption("Click any query below to run the evaluation test:")

    sample_queries = [
        ("🧠", "What is Agentic AI according to the eBook?"),
        ("⚡", "How do AI agents differ from traditional automation systems?"),
        ("🏛️", "What are the core components of an Agentic Architecture?"),
        ("💾", "What role does memory play in Agentic AI workflows?"),
        ("⚽", "Who won the 2022 FIFA World Cup?"),
    ]

    for icon, sq in sample_queries:
        if st.button(f"{icon} {sq}", key=f"btn_{sq}", use_container_width=True):
            st.session_state["user_input_val"] = sq

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieved_context = []
        st.session_state.last_score = None
        st.session_state.last_query = None
        st.rerun()

    st.caption("Developed for Appening AI Assessment • Python 3.13 • LangGraph")


# Cache compiled graph
@st.cache_resource
def load_graph():
    return build_rag_graph()


# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved_context" not in st.session_state:
    st.session_state.last_retrieved_context = []
if "last_score" not in st.session_state:
    st.session_state.last_score = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# Main Layout Split: Left (Chat Conversation) - Right (Inspector)
col1, col2 = st.columns([7, 5], gap="large")

with col1:
    st.markdown('### 💬 Dialogue Console')

    # Display empty state hero if no messages yet
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-title">Ready for Strict Document Exploration</div>
                <div class="welcome-desc">
                    Ask any question about Agentic AI, multi-agent workflows, perception, or memory. 
                    The chatbot answers strictly from the eBook context and will refuse out-of-domain questions.
                </div>
                <div class="feature-grid">
                    <div class="feature-item">
                        <div class="feature-item-title">🎯 Strict Grounding</div>
                        <div class="feature-item-desc">Answers are extracted strictly from 119 eBook text chunks.</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-item-title">🛡️ Anti-Hallucination</div>
                        <div class="feature-item-desc">Refusal guard explicitly halts queries outside document scope.</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-item-title">📊 Live Inspector</div>
                        <div class="feature-item-desc">Real-time transparency into chunks and similarity scoring.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚡"):
            st.markdown(msg["content"])

    # Chat input box
    user_query = st.chat_input("Ask a question about the Agentic AI eBook...")

    # Check if a benchmark button was clicked
    if "user_input_val" in st.session_state and st.session_state["user_input_val"]:
        user_query = st.session_state["user_input_val"]
        st.session_state["user_input_val"] = None

    if user_query:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.last_query = user_query

        # Rerun to render user message immediately
        st.rerun()

    # Process response if user just asked a question and assistant hasn't replied yet
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_question = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Retrieving document chunks and executing LangGraph workflow..."):
                try:
                    graph = load_graph()
                    result = graph.invoke({
                        "question": latest_question,
                        "context": [],
                        "answer": "",
                        "score": 0.0,
                    })

                    answer = result["answer"]
                    context = result["context"]
                    score = result["score"]

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_retrieved_context = context
                    st.session_state.last_score = score
                    st.rerun()
                except Exception as e:
                    err_msg = f"⚠️ Workflow Error: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})


with col2:
    st.markdown('<div class="inspector-header">🔍 Context & Grounding Inspector</div>', unsafe_allow_html=True)

    if st.session_state.last_score is not None:
        confidence = float(st.session_state.last_score)
        pct = int(confidence * 100)

        if confidence >= 0.80:
            card_class = "score-high"
            status_text = "🟢 HIGH GROUNDING • FACTUAL"
            status_color = "#34d399"
            fill_gradient = "linear-gradient(90deg, #10b981 0%, #34d399 100%)"
            explanation = "Answer is strictly substantiated by top retrieved document passages."
        elif confidence > 0.0:
            card_class = "score-medium"
            status_text = "🟡 MODERATE GROUNDING"
            status_color = "#fbbf24"
            fill_gradient = "linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%)"
            explanation = "Answer synthesized with partial document similarity match."
        else:
            card_class = "score-refusal"
            status_text = "🔴 OUT-OF-DOMAIN • REFUSAL TRIGGERED"
            status_color = "#f87171"
            fill_gradient = "linear-gradient(90deg, #ef4444 0%, #f87171 100%)"
            explanation = "Query has no semantic match in the eBook. System refused to hallucinate."

        st.markdown(
            f"""
            <div class="score-card-v2 {card_class}">
                <div class="score-label" style="color: {status_color};">{status_text}</div>
                <div class="score-number" style="color: {status_color};">{confidence:.4f} <span style="font-size: 1.1rem; color: #64748b; font-weight: 500;">/ 1.0</span></div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {pct}%; background: {fill_gradient};"></div>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.last_retrieved_context:
        num_chunks = len(st.session_state.last_retrieved_context)
        st.markdown(f"**Retrieved Context Passages ({num_chunks})**")

        for idx, chunk_text in enumerate(st.session_state.last_retrieved_context, 1):
            clean_text = chunk_text.strip()
            word_count = len(clean_text.split())
            st.markdown(
                f"""
                <div class="chunk-box">
                    <div class="chunk-meta">
                        <span>📄 Passage Chunk #{idx}</span>
                        <span>~{word_count} Words</span>
                    </div>
                    <div class="chunk-content">{clean_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 24px; text-align: center; color: #64748b;">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">📑</div>
                <div style="font-weight: 600; color: #94a3b8; margin-bottom: 4px;">No Active Retrieval Query</div>
                <div style="font-size: 0.82rem;">Select a benchmark query or enter a question to inspect retrieved context chunks in real time.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
