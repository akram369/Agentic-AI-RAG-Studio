import streamlit as st
import os
from src.config import PINECONE_INDEX_NAME, OPENAI_API_KEY, PINECONE_API_KEY
from src.graph import build_rag_graph

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Custom Styling for modern look
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .score-card {
        padding: 12px 18px;
        background-color: #f1f5f9;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        margin-bottom: 1rem;
    }
    .chunk-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-header">🤖 Agentic AI RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Strictly grounded Q&A on the Agentic AI eBook using LangGraph & Pinecone</div>',
    unsafe_allow_html=True,
)

# Sidebar with status and configuration
with st.sidebar:
    st.header("⚙️ Configuration & Status")
    st.write(f"**Index Name:** `{PINECONE_INDEX_NAME}`")
    st.write(f"**OpenAI API Key:** {'✅ Set' if OPENAI_API_KEY else '❌ Missing'}")
    st.write(f"**Pinecone API Key:** {'✅ Set' if PINECONE_API_KEY else '❌ Missing'}")

    st.markdown("---")
    st.subheader("💡 Benchmark Queries")
    sample_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?",
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}"):
            st.session_state["user_input_val"] = sq

    is_live = bool(OPENAI_API_KEY) and bool(PINECONE_API_KEY) and not os.getenv("MOCK_MODE", "").lower() in ("true", "1")
    st.write(f"**Execution Mode:** {'🟢 Live (Pinecone + OpenAI)' if is_live else '🟡 Local Audit / Fallback'}")

    st.markdown("---")
    st.info("ℹ️ Powered by LangGraph StateGraph, OpenAI `gpt-4o-mini`, `text-embedding-3-small`, and Pinecone.")


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

# Layout split: Chat conversation and Context Inspector
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💬 Chat")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input handling
    user_query = st.chat_input("Ask a question about Agentic AI...")

    # Check if sample query button was clicked
    if "user_input_val" in st.session_state and st.session_state["user_input_val"]:
        user_query = st.session_state["user_input_val"]
        st.session_state["user_input_val"] = None

    if user_query:
        # User message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant processing
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating grounded response..."):
                try:
                    graph = load_graph()
                    result = graph.invoke({
                        "question": user_query,
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
                except Exception as e:
                    err_text = f"An error occurred: {str(e)}"
                    st.error(err_text)
                    st.session_state.messages.append({"role": "assistant", "content": err_text})

with col2:
    st.subheader("🔍 Context & Grounding Inspector")
    if st.session_state.last_score is not None:
        confidence = st.session_state.last_score
        st.markdown(
            f"""
            <div class="score-card">
                <strong>Relevance / Confidence Score:</strong> {confidence:.4f} / 1.0<br/>
                <small>{'🟢 High Grounding' if confidence > 0.8 else ('🟡 Moderate Grounding' if confidence > 0.4 else '🔴 Unrelated / Refusal')}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.last_retrieved_context:
        st.write(f"**Retrieved Chunks ({len(st.session_state.last_retrieved_context)}):**")
        for i, chunk in enumerate(st.session_state.last_retrieved_context, 1):
            with st.expander(f"📄 Chunk {i}", expanded=(i == 1)):
                st.text(chunk)
    else:
        st.info("Ask a question to inspect retrieved context chunks and confidence scoring in real time.")
