# 📘 CRL Review Agent - Usage Guide

Complete guide to using the AI-powered document review system.

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Using the Streamlit UI](#using-the-streamlit-ui)
4. [Using the CLI](#using-the-cli)
5. [Using the Python API](#using-the-python-api)
6. [Configuration Options](#configuration-options)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### Step 1: Set up Environment

```bash
# Navigate to project directory
cd /Users/raghavrama/Desktop/nvidia-hackathon

# Activate virtual environment (if you have one)
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get NVIDIA API Key

1. Go to [build.nvidia.com](https://build.nvidia.com/)
2. Sign up/Sign in
3. Get your free API key
4. Set it as environment variable:

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

Or create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your key
```

---

## ⚡ Quick Start

### Easiest Way: Use the Quick Start Script

```bash
./quick_start.sh
```

This will:
- Activate the virtual environment
- Check/install dependencies
- Prompt for API key if not set
- Launch the Streamlit UI

### Manual Start

**Option 1: Streamlit UI**
```bash
streamlit run streamlit_review_app.py
```

**Option 2: CLI**
```bash
python review_cli.py review ./unapproved_CRLs/CRL_BLA125745_20250115.pdf
```

**Option 3: Python Script**
```bash
python crl_review_agent.py
```

---

## 🎨 Using the Streamlit UI

### First Time Setup

1. **Launch the app:**
   ```bash
   streamlit run streamlit_review_app.py
   ```

2. **Enter API Key:**
   - Look at the sidebar
   - Enter your NVIDIA API key
   - The key will be saved for your session

3. **Build Knowledge Base** (first time only):
   - Click "Build Knowledge Base" button
   - Wait 5-10 minutes for 202 PDFs to be processed
   - The vectorstore will be saved to disk

   **Note:** You only need to do this once! Next time, use "Load Existing Knowledge Base"

### Reviewing a Document

1. **Initialize Agent:**
   - If first time: Click "Build Knowledge Base"
   - If returning: Click "Load Existing Knowledge Base"
   - Wait for "Agent Ready" message

2. **Select Document:**
   - Use dropdown to select from `unapproved_CRLs/`
   - Or upload a new PDF file

3. **Start Review:**
   - Click "🔍 Review Document"
   - Wait 1-2 minutes for analysis
   - View comprehensive report

4. **Download Report:**
   - Click "📥 Download Review Report"
   - Saves as text file

### Batch Review Mode

1. Go to "📊 Batch Review" tab
2. Select multiple documents from the list
3. Click "🚀 Start Batch Review"
4. Wait for all reviews to complete
5. Expand each result to view

### UI Tips

- **Sidebar**: Contains all configuration options
- **Model Selection**: Change LLM/embedding models on the fly
- **Advanced Settings**: Adjust chunk size and overlap
- **Metrics**: Shows count of approved/unapproved CRLs

---

## 💻 Using the CLI

The CLI is perfect for automation, scripting, and quick reviews.

### Build Knowledge Base

```bash
python review_cli.py build
```

With options:
```bash
python review_cli.py build \
  --approved-dir ./approved_CRLs \
  --save-path ./vectorstore
```

### Review Single Document

Basic:
```bash
python review_cli.py review ./unapproved_CRLs/CRL_BLA125745_20250115.pdf
```

With options:
```bash
python review_cli.py review ./unapproved_CRLs/CRL_BLA125745_20250115.pdf \
  --model meta/llama-3.1-405b-instruct \
  --output review_report.txt \
  --load-kb ./vectorstore
```

### Batch Review

Review all documents in a directory:
```bash
python review_cli.py batch ./unapproved_CRLs/
```

With options:
```bash
python review_cli.py batch ./unapproved_CRLs/ \
  --model meta/llama-3.1-70b-instruct \
  --output-dir ./reviews \
  --load-kb ./vectorstore
```

### CLI Tips

- Results are printed to console and optionally saved to files
- Use `--help` for any command to see all options
- Progress is shown for batch operations
- Errors are clearly reported with suggestions

---

## 🐍 Using the Python API

For integration into your own scripts and applications.

### Basic Usage

```python
import os
from crl_review_agent import CRLReviewAgent

# Set API key (if not in environment)
os.environ["NVIDIA_API_KEY"] = "your-key-here"

# Initialize agent
agent = CRLReviewAgent(
    approved_crls_dir="./approved_CRLs",
    llm_model="meta/llama-3.1-70b-instruct"
)

# Load knowledge base (or build if first time)
try:
    agent.load_knowledge_base("./vectorstore")
except:
    agent.build_knowledge_base()
    agent.save_knowledge_base("./vectorstore")

# Create the agent
agent.create_agent()

# Review a document
result = agent.review_document("./unapproved_CRLs/CRL_BLA125745_20250115.pdf")

# Access results
print(f"Document: {result['document_path']}")
print(f"Pages: {result['document_pages']}")
print(f"Review:\n{result['review']}")
```

### Advanced Usage

```python
# Custom configuration
agent = CRLReviewAgent(
    approved_crls_dir="./approved_CRLs",
    chunk_size=1500,
    chunk_overlap=300,
    llm_model="meta/llama-3.1-405b-instruct",
    embedding_model="nvidia/nv-embedqa-e5-v5",
    rerank_model="nvidia/llama-3.2-nv-rerankqa-1b-v2"
)

# Build knowledge base with custom settings
agent.build_knowledge_base()
agent.save_knowledge_base("./custom_vectorstore")

# Batch review
documents = [
    "./unapproved_CRLs/CRL_BLA125745_20250115.pdf",
    "./unapproved_CRLs/CRL_NDA205508_20250728.pdf"
]
results = agent.batch_review(documents)

# Process results
for result in results:
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Review for {result['document_path']}")
        print(result['review'])
        print("\n" + "="*80 + "\n")
```

### Integration Example

```python
# Example: Automated review workflow
import glob
from pathlib import Path

def automated_review_workflow():
    """Review all pending CRLs and generate reports"""
    
    # Setup
    agent = CRLReviewAgent()
    agent.load_knowledge_base("./vectorstore")
    agent.create_agent()
    
    # Find all pending documents
    pending_docs = glob.glob("./unapproved_CRLs/*.pdf")
    
    # Review each and save report
    reports_dir = Path("./reports")
    reports_dir.mkdir(exist_ok=True)
    
    for doc_path in pending_docs:
        print(f"Processing: {doc_path}")
        
        # Review
        result = agent.review_document(doc_path)
        
        # Save report
        doc_name = Path(doc_path).stem
        report_path = reports_dir / f"{doc_name}_review.txt"
        
        with open(report_path, 'w') as f:
            f.write(result['review'])
        
        print(f"✅ Saved report to {report_path}")
    
    print(f"\n✅ All reviews complete! {len(pending_docs)} documents processed.")

# Run workflow
automated_review_workflow()
```

---

## ⚙️ Configuration Options

### Agent Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `approved_crls_dir` | `"./approved_CRLs"` | Directory with reference CRLs |
| `chunk_size` | `1000` | Size of text chunks (characters) |
| `chunk_overlap` | `200` | Overlap between chunks |
| `llm_model` | `"meta/llama-3.1-70b-instruct"` | LLM for reasoning |
| `embedding_model` | `"nvidia/llama-3.2-nv-embedqa-1b-v2"` | Embedding model |
| `rerank_model` | `"nvidia/llama-3.2-nv-rerankqa-1b-v2"` | Reranking model |

### Available Models

**LLM Models (for reasoning):**
- `meta/llama-3.1-70b-instruct` - Best balance (default)
- `meta/llama-3.1-405b-instruct` - Highest quality
- `nvidia/nvidia-nemotron-nano-9b-v2` - Fast, good for testing
- `mistralai/mixtral-8x7b-instruct-v0.1` - Alternative

**Embedding Models:**
- `nvidia/llama-3.2-nv-embedqa-1b-v2` - Good balance (default)
- `nvidia/nv-embedqa-e5-v5` - Higher quality

**Reranking Models:**
- `nvidia/llama-3.2-nv-rerankqa-1b-v2` - Best for CRL documents

### Retrieval Configuration

Modify retrieval behavior:

```python
# In crl_review_agent.py, adjust these in build_knowledge_base():
base_retriever = self.vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 10}    # number of chunks to retrieve
)

reranker = NVIDIARerank(
    model=self.rerank_model,
    top_n=5  # final number after reranking
)
```

---

## 💡 Best Practices

### For Best Results

1. **Use Llama 3.1 70B** for production reviews
2. **Use Nemotron Nano 9B** for quick testing/iteration
3. **Chunk size 1000-1500** works well for CRLs
4. **Save vectorstore** after building (saves time)
5. **Review in batches** overnight for large sets

### Document Preparation

- Ensure PDFs are text-based (not scanned images)
- Check that PDFs are not password-protected
- Remove any corrupted or incomplete files

### Performance Optimization

1. **First time setup:**
   - Build vectorstore once: ~10 minutes
   - Save to disk: reuse in future sessions

2. **Review speed:**
   - Single document: 1-2 minutes
   - Batch of 10: ~15-20 minutes
   - Use smaller model for faster results

3. **Memory usage:**
   - 202 PDFs vectorstore: ~500MB RAM
   - Use `faiss-cpu` if memory constrained

### Quality Tips

1. **Model selection:**
   - Critical reviews: Use Llama 3.1 405B
   - Standard reviews: Use Llama 3.1 70B
   - Quick checks: Use Nemotron Nano 9B

2. **Chunk size:**
   - Larger (1500): More context, slower
   - Smaller (800): Faster, less context
   - Sweet spot: 1000-1200

3. **Retrieval:**
   - More chunks (k=15): More comprehensive, slower
   - Fewer chunks (k=5): Faster, may miss context
   - Recommended: k=10, top_n=5 after rerank

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "NVIDIA_API_KEY not set"

**Solution:**
```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

Or add to `.env` file:
```
NVIDIA_API_KEY=nvapi-your-key-here
```

---

#### Issue: "Knowledge base not found"

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: './vectorstore'`

**Solution:**
Build the knowledge base first:
```bash
python review_cli.py build
```

Or in Python:
```python
agent.build_knowledge_base()
agent.save_knowledge_base("./vectorstore")
```

---

#### Issue: "Rate limit exceeded"

**Error:** `429 Too Many Requests`

**Solutions:**
1. Wait 30-60 seconds and retry
2. Reduce batch size
3. Add delays between requests
4. Upgrade your NVIDIA API tier

---

#### Issue: "Out of memory"

**Error:** `MemoryError` or system slowdown

**Solutions:**
1. Reduce chunk_size to 800
2. Reduce k value to 5
3. Close other applications
4. Use `faiss-cpu` instead of `faiss-gpu`

---

#### Issue: "PDF loading failed"

**Error:** `PyPDFLoader error`

**Solutions:**
1. Check if PDF is corrupted
2. Ensure PDF is not password-protected
3. Verify file path is correct
4. Try re-downloading the PDF

---

#### Issue: "Slow performance"

**Symptoms:** Reviews taking >5 minutes

**Solutions:**
1. Use faster model (Nemotron Nano 9B)
2. Reduce chunk_overlap to 100
3. Reduce retrieval k value to 5
4. Check internet connection speed

---

#### Issue: "Agent gives generic responses"

**Symptoms:** Review doesn't cite approved CRLs

**Solutions:**
1. Ensure knowledge base was built correctly
2. Try rebuilding with smaller chunk_size
3. Increase k value for more retrieval
4. Check that approved_CRLs directory has PDFs

---

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your code
agent = CRLReviewAgent()
# ... rest of code
```

For CLI:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_PROJECT="crl-review"
```

---

## 📞 Getting Help

1. **Check this guide** for common solutions
2. **Review README.md** for overview and features
3. **Check NVIDIA docs**: [docs.nvidia.com](https://docs.nvidia.com)
4. **LangChain docs**: [python.langchain.com](https://python.langchain.com)
5. **GitHub issues** for bug reports

---

## 📝 Example Workflows

### Workflow 1: Daily Review Process

```bash
# Morning: Build/update knowledge base if needed
python review_cli.py build

# Review today's submissions
python review_cli.py batch ./unapproved_CRLs/today/ \
  --output-dir ./reviews/$(date +%Y-%m-%d)

# Check results
ls ./reviews/$(date +%Y-%m-%d)/
```

### Workflow 2: Interactive Review

```bash
# Start UI
streamlit run streamlit_review_app.py

# 1. Load knowledge base (sidebar)
# 2. Select document from dropdown
# 3. Click "Review Document"
# 4. Read analysis
# 5. Download report
```

### Workflow 3: Automated Pipeline

```python
# pipeline.py
from crl_review_agent import CRLReviewAgent
from pathlib import Path
import json

def review_pipeline():
    # Setup
    agent = CRLReviewAgent()
    agent.load_knowledge_base()
    agent.create_agent()
    
    # Find new documents
    pending = Path("./unapproved_CRLs").glob("*.pdf")
    
    # Review each
    results = []
    for doc in pending:
        result = agent.review_document(str(doc))
        results.append({
            "document": doc.name,
            "pages": result['document_pages'],
            "review": result['review']
        })
    
    # Save summary
    with open("daily_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Reviewed {len(results)} documents")

if __name__ == "__main__":
    review_pipeline()
```

---

**Happy Reviewing! 🚀**

For questions or issues, check the Troubleshooting section or consult the README.md.

