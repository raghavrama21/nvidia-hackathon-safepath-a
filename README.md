# 🧠 SafePath AI - Autonomous Regulatory Risk Intelligence Agent

[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-76B900?style=for-the-badge&logo=nvidia)](https://build.nvidia.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green?style=for-the-badge)](https://www.langchain.com/)

An **autonomous AI agent** that analyzes FDA regulatory documents (CRLs, NDAs, BLAs) using multi-step reasoning, retrieval + reranking, and domain-specific risk intelligence. Built with NVIDIA NIM, LangChain, and LangGraph.

🎯 **Demonstrates**: Autonomous reasoning • Multi-step workflows • Advanced RAG with reranking • Real-world pharma applicability

---

## ✨ What is SafePath AI?

SafePath AI is an autonomous regulatory intelligence agent that:
- 🤖 **Autonomously analyzes** regulatory documents through 6-step workflows
- 🔄 **Retrieves + Reranks** using NVIDIA Nemotron models for maximum relevance
- 🎯 **Scores risks** across 6 regulatory domains (CMC, Clinical, Nonclinical, etc.)
- ✅ **Generates actions** with automated checklists, JIRA-style tickets
- 💰 **Projects costs** and timelines for remediation with ROI calculations

## 🎯 Features

- **Risk Scoring**: Automated 1-10 risk assessment for document approval likelihood
- **Deficiency Detection**: Automatically identifies and highlights critical deficiencies
- **Intelligent Document Review**: AI agent analyzes CRLs against 291 reference documents
- **Interactive Chat**: Ask questions about FDA CRLs and regulatory requirements
- **Multi-Model Support**: Uses NVIDIA NIM models (Nemotron, Llama, etc.)
- **Advanced RAG**: Retrieval-Augmented Generation with reranking for accurate context
- **Beautiful UI**: Streamlit-based web interface for easy interaction
- **Batch Processing**: Review multiple documents at once
- **Agentic Reasoning**: Uses ReAct framework for step-by-step analysis

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- NVIDIA API key (get free at [build.nvidia.com](https://build.nvidia.com/))

### 2. Installation

```bash
# Activate your virtual environment (if you have one)
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set up your API key

**⚠️ IMPORTANT: Never commit your API key to GitHub!**

Create a `.env` file in the project root (this file is git-ignored):

```bash
# Create .env file
echo "NVIDIA_API_KEY=your-nvidia-api-key-here" > .env
```

Get your free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com/)

The API key will be automatically loaded from `.env` - no need to export it every time!

### 4. Build Knowledge Base (One-Time Setup - 3 minutes)

**⚡ IMPORTANT: Build once, load instantly forever!**

```bash
# Run this ONCE to build and save the knowledge base
python build_kb_once.py
```

This processes 291 PDFs and saves the embeddings. You only need to do this:
- ✅ Once initially
- ✅ When you add new CRLs
- ✅ When you want to change chunking parameters

After building, the knowledge base will load in **5-10 seconds** instead of **3 minutes**!

### 5. Run the application

**Option A: Streamlit UI (Recommended)**
```bash
streamlit run streamlit_review_app.py

# Then click "⚡ Quick Load" to load the saved knowledge base instantly
```

**Option B: CLI Tool**
```bash
# Chat with the agent
python review_cli.py chat

# Review a document
python review_cli.py review ./demo_NDAs/SyntheticNDA_ChatGPT.pdf

# Batch review
python review_cli.py batch ./demo_NDAs/

# Rebuild knowledge base (if needed)
python review_cli.py build
```

**Option C: Quick Start Script**
```bash
./quick_start.sh
```

## 📁 Project Structure

```
nvidia-hackathon/
├── approved_CRLs/          # 202 approved CRL PDFs (knowledge base)
├── unapproved_CRLs/        # 89 unapproved CRLs PDFs (knowledge base)
├── demo_NDAs/              # Testing the agent
├── crl_review_agent.py     # Core agent implementation
├── streamlit_review_app.py # Web UI (SafePath)
├── review_cli.py           # CLI tool for automation
├── quick_start.sh          # Easy launcher script
├── requirements.txt        # Dependencies
├── .env                    # API key configuration
├── vectorstore/            # FAISS vector database (generated)
└── README.md
```

## 🛠️ How It Works

### Architecture

```
┌─────────────────────┐
│  Approved CRLs      │
│  (202 PDFs)         │
│                     │
│  Unapproved CRLs    │
│  (89 PDFs)          │
└──────────┬──────────┘
           │
           │ Chunk & Embed
           ↓
┌─────────────────────┐
│  FAISS Vector DB    │
│  (Knowledge Base)   │
└──────────┬──────────┘
           │
           │ Retrieve
           ↓
┌─────────────────────┐      ┌──────────────────┐
│  Document to Review │─────→│   AI Agent       │
│  (Upload CRL)       │      │  (LangGraph)     │
└─────────────────────┘      └────────┬─────────┘
                                      │
                                      │ Analyze
                                      ↓
                             ┌──────────────────┐
                             │  Review Report   │
                             │  - Compliance    │
                             │  - Issues        │
                             │  - Recommendations│
                             └──────────────────┘
```

### Process Flow

1. **Knowledge Base Creation**: 
   - Load all approved CRL PDFs
   - Split into chunks (1000 tokens, 200 overlap)
   - Create embeddings using NVIDIA NV-EmbedQA
   - Store in FAISS vector database

2. **Document Review**:
   - User uploads/selects a CRL to review
   - Agent receives the document and review instructions
   - Agent uses retrieval tool to find relevant context from approved CRLs
   - Agent performs multi-step reasoning (ReAct framework)
   - Agent generates comprehensive review report

3. **Output**:
   - **Risk Score**: Numerical 1-10 rating (1-3: High Risk, 4-6: Medium, 7-9: Low, 10: Minimal)
   - **Critical Deficiencies**: Structured list of specific issues found
   - Overall assessment
   - Compliance analysis
   - Identified strengths
   - Prioritized recommendations (Critical/Important/Minor)
   - Approval likelihood

## 🤖 Available Models

### LLM Models (Reasoning)
- `nvidia/nvidia-nemotron-nano-9b-v2` ⭐ (default, fast & efficient)
- `meta/llama-3.1-70b-instruct` (higher quality)
- `meta/llama-3.1-405b-instruct` (highest quality, slower)
- `mistralai/mixtral-8x7b-instruct-v0.1`

### Embedding Models  
- `nvidia/nv-embedqa-e5-v5` ⭐ (default, best for PDFs with tables/images)
- `nvidia/llama-3.2-nv-embedqa-1b-v2` (alternative)

### Reranking Models
- `nvidia/llama-3.2-nv-rerankqa-1b-v2` ⭐ (default)

## 📖 Usage Examples

### Streamlit UI (SafePath)

1. **Initialize Agent**:
   - API key is automatically loaded from `.env` file
   - Click "Load Existing Knowledge Base" or "Build Knowledge Base"
   - Wait for initialization (building takes ~5-10 minutes for 291 PDFs)

2. **Review Single Document**:
   - Select a CRL from the dropdown
   - Or upload a new PDF
   - Click "Review Document"
   - View comprehensive analysis

3. **Chat with Agent**:
   - Go to "Chat with Agent" tab
   - Ask questions about FDA CRLs, regulations, common issues
   - Get answers grounded in 291 CRLs
   - Build conversation history

4. **Batch Review**:
   - Go to "Batch Review" tab
   - Select multiple documents
   - Click "Start Batch Review"
   - Download results

### CLI Tool

```bash
# Build knowledge base (first time)
python review_cli.py build

# Chat with the agent
python review_cli.py chat
# Ask questions like:
# - "What are common CMC deficiencies in approved CRLs?"
# - "What sections should a CRL contain?"
# - "How should clinical data be presented?"

# Review a single document
python review_cli.py review ./demo_NDAs/SyntheticNDA_ChatGPT.pdf

# Review all documents in a directory
python review_cli.py batch ./demo_NDAs/ --output-dir ./reviews

# Use different model
python review_cli.py review SyntheticNDA_ChatGPT.pdf --model meta/llama-3.1-70b-instruct
```

### Python API

```python
from crl_review_agent import CRLReviewAgent

# Initialize (API key loaded from .env automatically)
agent = CRLReviewAgent(approved_crls_dir="./approved_CRLs")

# Load knowledge base (or build if first time)
agent.load_knowledge_base("./vectorstore")
agent.create_agent()

# Chat with the agent
response = agent.chat("What are common CMC deficiencies in approved CRLs?")
print(response)

# Review a document (returns structured results)
result = agent.review_document("./demo_NDAs/SyntheticNDA_ChatGPT.pdf")

# Access risk score
print(f"Risk Score: {result['risk_score']['score']}/10")
print(f"Risk Category: {result['risk_score']['category']}")

# Access deficiencies
print(f"\nDeficiencies found: {len(result['deficiencies'])}")
for i, deficiency in enumerate(result['deficiencies'], 1):
    print(f"{i}. {deficiency}")

# Full review text
print(f"\nFull Review:\n{result['review']}")
```

## 🎓 Use Cases

- **Pre-submission Review**: Check documents before submitting to FDA
- **Quality Assurance**: Ensure compliance with regulatory standards
- **Training**: Learn from approved examples
- **Risk Assessment**: Evaluate likelihood of approval
- **Pattern Recognition**: Identify common issues and best practices

## 🔧 Configuration

### Adjust Chunk Size
```python
agent = CRLReviewAgent(
    chunk_size=1500,        # Larger chunks = more context
    chunk_overlap=300,      # More overlap = better continuity
)
```

### Change Models
```python
agent = CRLReviewAgent(
    llm_model="meta/llama-3.1-405b-instruct",  # Use higher quality model
    embedding_model="nvidia/nv-embedqa-e5-v5",  # Best for PDFs with tables/images
    chunk_size=1500,  # Larger chunks
)
```

## 📊 Dataset Information

- **Letters**: 291 approved & unapproved FDA Complete Response Letters
- **Document Types**: BLA (Biologics License Application) and NDA (New Drug Application)

## 🚨 Troubleshooting

### Issue: "Knowledge base not found"
**Solution**: Run with "Build Knowledge Base" option first

### Issue: "NVIDIA_API_KEY not set"
**Solution**: 
```bash
# Create .env file with your API key
echo "NVIDIA_API_KEY=nvapi-your-key-here" > .env

# The app will automatically load it
```

### Issue: "Rate limit exceeded"
**Solution**: 
- Wait a few moments and retry
- Consider using a smaller batch size
- Upgrade your NVIDIA API tier

### Issue: "Out of memory"
**Solution**:
- Reduce chunk_size (try 800)
- Reduce search_kwargs k value (try 5)
- Use faiss-cpu instead of faiss-gpu

## 🌟 Advanced Features

### Custom Review Prompts

```python
custom_prompt = """
Focus on:
1. Clinical safety data completeness
2. Manufacturing process validation
3. Labeling accuracy
"""

result = agent.review_document(doc_path, custom_instructions=custom_prompt)
```

### Export to Different Formats

```python
# Save as JSON
import json
with open('review.json', 'w') as f:
    json.dump(result, f, indent=2)

# Save as Markdown
with open('review.md', 'w') as f:
    f.write(f"# Review of {result['document_path']}\n\n")
    f.write(result['review'])
```

## 🤝 Contributing

This project was built for the NVIDIA Hackathon. Feel free to:
- Submit issues
- Propose new features
- Improve documentation
- Add new models

## 📄 License

MIT License - feel free to use and modify

## 🙏 Acknowledgments

- **NVIDIA NIM**: Powerful AI models and inference
- **LangChain**: Framework for LLM applications
- **LangGraph**: Agentic workflows
- **Streamlit**: Beautiful web interface

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review NVIDIA NIM documentation: [docs.nvidia.com](https://docs.nvidia.com)
3. LangChain docs: [python.langchain.com](https://python.langchain.com)

---

**Built with ❤️ for NVIDIA Hackathon 🚀**

**SafePath** - Powered by NVIDIA NIM | LangChain | LangGraph | Streamlit

