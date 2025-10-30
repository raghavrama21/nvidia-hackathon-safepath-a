"""
Streamlit UI for CRL Review Agent
Beautiful interface for document review using AI
"""

import os

# Fix for OpenMP on macOS (must be before other imports)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from pathlib import Path
from crl_review_agent import CRLReviewAgent
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SafePath - AI CRL Review",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #76B900;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    .stButton>button {
        width: 100%;
        background-color: #76B900;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #5a9100;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🛡️ SafePath</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered FDA Document Review using NVIDIA NIM</div>',
    unsafe_allow_html=True,
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "review_results" not in st.session_state:
    st.session_state.review_results = None

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key - automatically loaded from .env file
    api_key = os.getenv("NVIDIA_API_KEY", "")

    if api_key:
        st.success("✅ API Key loaded from .env file")
    else:
        st.warning("⚠️ No API key found in .env file")
        st.info("Create a .env file with: NVIDIA_API_KEY=your-key-here")

    # Optional: Allow override
    with st.expander("🔑 Override API Key (Optional)"):
        override_key = st.text_input(
            "Enter different API Key",
            type="password",
            help="Leave empty to use key from .env file",
        )
        if override_key:
            api_key = override_key
            os.environ["NVIDIA_API_KEY"] = override_key
            st.info("Using override key for this session")

    st.divider()

    # Model selection
    st.subheader("🤖 Model Settings")
    llm_model = st.selectbox(
        "LLM Model",
        [
            "nvidia/nvidia-nemotron-nano-9b-v2",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
        ],
        help="Primary reasoning model",
    )

    embedding_model = st.selectbox(
        "Embedding Model",
        [
            "nvidia/nv-embedqa-e5-v5",
            "nvidia/llama-3.2-nv-embedqa-1b-v2",
        ],
        help="Model for document embeddings (E5-v5 is best for PDFs with tables/images)",
    )

    st.divider()

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)

    st.divider()

    # Knowledge base stats
    approved_dir = Path("./approved_CRLs")
    unapproved_dir = Path("./unapproved_CRLs")

    if approved_dir.exists():
        approved_count = len(list(approved_dir.glob("*.pdf")))
        st.metric("Approved CRLs", approved_count)

    if unapproved_dir.exists():
        unapproved_count = len(list(unapproved_dir.glob("*.pdf")))
        st.metric("Unapproved CRLs", unapproved_count)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(
    ["🚀 Setup & Review", "💬 Chat with Agent", "📊 Batch Review", "ℹ️ About"]
)

with tab1:
    # Step 1: Initialize Agent
    st.header("Step 1: Initialize Agent")

    # Check if saved knowledge base exists
    kb_exists = os.path.exists("./vectorstore")
    if kb_exists and not st.session_state.kb_ready:
        st.info(
            "💡 Saved knowledge base found! Click 'Quick Load' to start instantly (5-10 seconds)."
        )

    if not api_key:
        st.markdown(
            """
        <div class="status-box warning-box">
            <strong>⚠️ API Key Required</strong><br>
            Please enter your NVIDIA API Key in the sidebar to get started.<br>
            Get your free key at <a href="https://build.nvidia.com/" target="_blank">build.nvidia.com</a>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            # Quick load button - prioritized if KB exists
            if st.button(
                "⚡ Quick Load" if kb_exists else "📥 Load KB (No File Found)",
                disabled=st.session_state.kb_ready or not kb_exists,
                type="primary" if kb_exists else "secondary",
            ):
                with st.spinner("⚡ Loading saved knowledge base... (fast)"):
                    try:
                        st.session_state.agent = CRLReviewAgent(
                            approved_crls_dir="./approved_CRLs",
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            llm_model=llm_model,
                            embedding_model=embedding_model,
                        )
                        st.session_state.agent.load_knowledge_base("./vectorstore")
                        st.session_state.agent.create_agent()
                        st.session_state.kb_ready = True
                        st.success("✅ Knowledge base loaded and agent ready!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading knowledge base: {e}")
                        st.info("Try building the knowledge base first.")

        with col2:
            if st.button("🏗️ Build New KB", disabled=st.session_state.kb_ready):
                with st.spinner(
                    "Building knowledge base from 202 approved CRLs... (~3 minutes, only once)"
                ):
                    try:
                        st.session_state.agent = CRLReviewAgent(
                            approved_crls_dir="./approved_CRLs",
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            llm_model=llm_model,
                            embedding_model=embedding_model,
                        )
                        st.session_state.agent.build_knowledge_base()
                        st.session_state.agent.save_knowledge_base("./vectorstore")
                        st.session_state.agent.create_agent()
                        st.session_state.kb_ready = True
                        st.success(
                            "✅ Knowledge base built, saved, and agent ready! Next time use Quick Load."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error building knowledge base: {e}")

        with col3:
            if st.button("🔄 Rebuild KB", disabled=st.session_state.kb_ready):
                if kb_exists:
                    confirm = st.warning(
                        "This will rebuild the knowledge base from scratch. Continue?"
                    )
                with st.spinner("Rebuilding knowledge base..."):
                    try:
                        st.session_state.agent = CRLReviewAgent(
                            approved_crls_dir="./approved_CRLs",
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            llm_model=llm_model,
                            embedding_model=embedding_model,
                        )
                        st.session_state.agent.build_knowledge_base()
                        st.session_state.agent.save_knowledge_base("./vectorstore")
                        st.session_state.agent.create_agent()
                        st.session_state.kb_ready = True
                        st.success("✅ Knowledge base rebuilt successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error rebuilding knowledge base: {e}")

    if st.session_state.kb_ready:
        st.markdown(
            """
        <div class="status-box success-box">
            <strong>✅ Agent Ready</strong><br>
            The AI agent is initialized and ready to review documents.
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Step 2: Select document to review
        st.header("Step 2: Select Document to Review")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Get list of unapproved CRLs
            unapproved_crls = sorted(list(Path("./unapproved_CRLs").glob("*.pdf")))

            if unapproved_crls:
                selected_doc = st.selectbox(
                    "Choose a CRL to review:",
                    unapproved_crls,
                    format_func=lambda x: x.name,
                )

                # Or upload a new document
                st.markdown("**Or upload a new document:**")
                uploaded_file = st.file_uploader(
                    "Upload CRL PDF", type=["pdf"], key="doc_upload"
                )

                if uploaded_file:
                    # Save uploaded file temporarily
                    temp_path = Path(f"./temp_{uploaded_file.name}")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    selected_doc = temp_path

                # Review button
                if st.button("🔍 Review Document", type="primary"):
                    with st.spinner(
                        "🤖 Agent is reviewing the document... This may take 1-2 minutes..."
                    ):
                        try:
                            result = st.session_state.agent.review_document(
                                str(selected_doc)
                            )
                            st.session_state.review_results = result
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during review: {e}")
            else:
                st.warning("No unapproved CRLs found in ./unapproved_CRLs directory")

        with col2:
            if unapproved_crls:
                st.info(
                    f"""
                **Document Info**
                - Total files: {len(unapproved_crls)}
                - Selected: {selected_doc.name if 'selected_doc' in locals() else 'None'}
                """
                )

        # Step 3: Display results
        if st.session_state.review_results:
            st.header("Step 3: Review Results")

            result = st.session_state.review_results

            # Risk Score - Display prominently at top
            if result.get("risk_score") and result["risk_score"]["score"] is not None:
                risk = result["risk_score"]

                # Color coding
                if risk["color"] == "green":
                    risk_color = "#28a745"
                elif risk["color"] == "orange":
                    risk_color = "#ffc107"
                else:
                    risk_color = "#dc3545"

                st.markdown(
                    f"""
                <div style="background-color: {risk_color}20; border-left: 5px solid {risk_color}; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
                    <h2 style="color: {risk_color}; margin: 0;">⚠️ Risk Score: {risk["score"]}/10</h2>
                    <h3 style="color: {risk_color}; margin: 5px 0;">{risk["category"]}</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Critical Deficiencies
            if result.get("deficiencies") and len(result["deficiencies"]) > 0:
                st.subheader("🚨 Critical Deficiencies Found")
                st.markdown(
                    f"**{len(result['deficiencies'])} deficiencies identified:**"
                )

                for i, deficiency in enumerate(result["deficiencies"], 1):
                    st.markdown(
                        f"""
                    <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 3px;">
                        <strong>{i}.</strong> {deficiency}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.divider()

            # Document info
            st.subheader("📄 Document Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File", Path(result["document_path"]).name)
            with col2:
                st.metric("Pages", result["document_pages"])
            with col3:
                st.metric("Agent Steps", result["agent_steps"])

            st.divider()

            # Review content
            st.subheader("📝 Detailed Review Analysis")
            st.markdown(result["review"])

            st.divider()

            # Download option
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.download_button(
                    label="📥 Download Review Report",
                    data=result["review"],
                    file_name=f"review_{Path(result['document_path']).stem}.txt",
                    mime="text/plain",
                )

with tab2:
    st.header("💬 Chat with Agent")

    if not st.session_state.kb_ready:
        st.info("Please initialize the agent in the 'Setup & Review' tab first.")
    else:
        st.markdown(
            """
        Ask questions about FDA CRLs, regulatory requirements, common issues, or anything related to 
        the 202 approved CRLs in the knowledge base.
        """
        )

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**🧑 You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 SafePath:** {message['content']}")
                st.markdown("---")

        # Chat input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_question = st.text_input(
                "Ask a question:",
                key="chat_input",
                placeholder="e.g., What are common CMC deficiencies in approved CRLs?",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            send_button = st.button("Send", type="primary", use_container_width=True)

        # Process question
        if send_button and user_question:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )

            # Get agent response
            with st.spinner("🤖 Agent is thinking..."):
                try:
                    # Create a query for the agent
                    query_prompt = f"""
Based on the knowledge base of approved FDA CRLs, please answer this question:

{user_question}

Provide a clear, concise answer with specific examples from the approved CRLs when relevant.
Cite sources using [CRL] when referencing specific documents.
"""

                    result = st.session_state.agent.agent.invoke(
                        {"messages": [("user", query_prompt)]}
                    )

                    # Extract response
                    response = result["messages"][-1].content

                    # Add agent response to history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )

                    # Rerun to update chat display
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

with tab3:
    st.header("📊 Batch Review Mode")

    if not st.session_state.kb_ready:
        st.info("Please initialize the agent in the 'Setup & Review' tab first.")
    else:
        st.markdown(
            """
        Review multiple documents at once. Select documents from the unapproved CRLs directory.
        """
        )

        unapproved_crls = sorted(list(Path("./unapproved_CRLs").glob("*.pdf")))

        if unapproved_crls:
            selected_docs = st.multiselect(
                "Select documents to review:",
                unapproved_crls,
                format_func=lambda x: x.name,
            )

            if selected_docs:
                st.info(f"Selected {len(selected_docs)} documents for batch review")

                if st.button("🚀 Start Batch Review"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []
                    for i, doc_path in enumerate(selected_docs):
                        status_text.text(
                            f"Reviewing {doc_path.name}... ({i+1}/{len(selected_docs)})"
                        )

                        try:
                            result = st.session_state.agent.review_document(
                                str(doc_path)
                            )
                            results.append(result)
                        except Exception as e:
                            st.error(f"Error reviewing {doc_path.name}: {e}")

                        progress_bar.progress((i + 1) / len(selected_docs))

                    status_text.text("✅ Batch review complete!")

                    # Display results
                    st.success(f"Completed review of {len(results)} documents")

                    for result in results:
                        with st.expander(f"📄 {Path(result['document_path']).name}"):
                            st.markdown(result["review"])
        else:
            st.warning("No unapproved CRLs found")

with tab4:
    st.header("ℹ️ About SafePath")

    st.markdown(
        """
    ### 🎯 Purpose
    SafePath is an AI agent that helps review FDA Complete Response Letters (CRLs) by comparing them against 
    a knowledge base of approved CRLs. It provides:
    
    - **Compliance Analysis**: Checks alignment with FDA standards
    - **Strength Identification**: Highlights what the document does well
    - **Issue Detection**: Identifies potential problems or inconsistencies
    - **Recommendations**: Provides actionable improvements
    - **Risk Assessment**: Evaluates likelihood of approval
    
    ### 🤖 Technology Stack
    - **LLM**: NVIDIA NIM (Llama 3.1, Nemotron, etc.)
    - **Embeddings**: NVIDIA NV-EmbedQA
    - **Reranking**: NVIDIA Llama 3.2 RerankQA
    - **Framework**: LangChain + LangGraph
    - **Vector Store**: FAISS
    - **UI**: Streamlit
    
    ### 📚 How It Works
    1. **Knowledge Base Creation**: All approved CRLs are processed, chunked, and embedded into a vector database
    2. **Document Upload**: You select or upload a CRL to review
    3. **AI Analysis**: The agent retrieves relevant context from approved CRLs and performs analysis
    4. **Report Generation**: Comprehensive review report with recommendations
    
    ### 🚀 Getting Started
    1. Get a free NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/)
    2. Enter your API key in the sidebar
    3. Build or load the knowledge base
    4. Select a document to review
    5. Get AI-powered insights!
    
    ### 📊 Dataset
    - **Approved CRLs**: 202 documents serving as reference knowledge
    - **Unapproved CRLs**: Documents awaiting review
    
    ### 🎓 Use Cases
    - Pre-submission document review
    - Quality assurance for regulatory submissions
    - Learning from approved examples
    - Risk assessment before submission
    - Training and education
    
    ---
    
    **Built for NVIDIA Hackathon** 🚀
    
    Powered by NVIDIA NIM and LangChain
    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>SafePath | Powered by NVIDIA NIM 🚀</p>
    <p>Built with Streamlit, LangChain, and LangGraph</p>
</div>
""",
    unsafe_allow_html=True,
)
