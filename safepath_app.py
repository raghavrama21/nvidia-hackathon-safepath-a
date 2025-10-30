import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import tempfile, os

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="SafePath RAG", layout="wide")
st.title("🧠 SafePath — Regulatory Risk Intelligence RAG")
st.markdown("Upload FDA CRLs or related documents to build your knowledge base.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more CRL PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            all_docs.extend(docs)

    st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDF(s).")

    # -------------------------------
    # Split Text into Chunks
    # -------------------------------
    st.write("📑 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    st.write(f"✅ Created {len(chunks)} chunks.")

    # -------------------------------
    # Create Embeddings + Vector DB
    # -------------------------------
    st.write("⚙️ Creating embeddings and FAISS vectorstore...")
    embeddings = OpenAIEmbeddings()  # requires OPENAI_API_KEY in your environment
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state["vs"] = vectorstore
    st.success("✅ Vector database created and ready for retrieval.")

# -------------------------------
# Query Interface
# -------------------------------
if "vs" in st.session_state:
    query = st.text_input("💬 Ask SafePath something about your uploaded CRLs:")
    if query:
        llm = OpenAI(temperature=0)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state["vs"].as_retriever(search_kwargs={"k": 3}),
        )
        with st.spinner("Thinking..."):
            answer = qa.run(query)
        st.markdown("### 🧩 Answer:")
        st.write(answer)
