# 🚀 Getting Started - CRL Review Agent

**5-Minute Quick Start Guide**

## What You Got

I've built you a complete AI document review system with:

✅ **CRL Review Agent** - AI that reviews FDA documents using 202 approved CRLs as reference  
✅ **Beautiful Streamlit UI** - Click-based interface for easy reviews  
✅ **CLI Tool** - Command-line interface for automation  
✅ **Python API** - Integrate into your own code  
✅ **Complete Documentation** - Everything you need to know  

---

## ⚡ Quick Start (Choose One)

### Option 1: Use the Easy Script (Recommended)

```bash
./quick_start.sh
```

That's it! The script will:
- Check your environment
- Ask for your API key
- Install dependencies
- Launch the UI

### Option 2: Manual Start

```bash
# 1. Set your API key
export NVIDIA_API_KEY="your-key-here"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the UI
streamlit run streamlit_review_app.py
```

---

## 🔑 Getting Your API Key

1. Go to **[build.nvidia.com](https://build.nvidia.com/)**
2. Sign up (it's free!)
3. Get your API key
4. Copy it - you'll need it!

---

## 📖 What to Do Next

### First Time Using the App

1. **Open the Streamlit UI** (from quick start above)
2. **Enter your API key** in the sidebar
3. **Click "Build Knowledge Base"**
   - This processes 202 approved CRLs
   - Takes ~10 minutes (one-time only!)
   - Creates a searchable database
4. **Select a document** from the dropdown
5. **Click "Review Document"**
6. **Read the AI's analysis!**

### Files You'll Use

| File | What It Does | When to Use |
|------|--------------|-------------|
| `streamlit_review_app.py` | Beautiful web interface | Most common - interactive reviews |
| `review_cli.py` | Command-line tool | Automation, scripting, batch jobs |
| `crl_review_agent.py` | Core Python API | Integrate into your own code |
| `quick_start.sh` | Easy launcher | First time setup |

### Documentation Files

| File | What's Inside |
|------|---------------|
| `README.md` | Overview and features |
| `USAGE_GUIDE.md` | Detailed how-to guide |
| `ARCHITECTURE.md` | Technical details |
| `GETTING_STARTED.md` | This file! |

---

## 💡 Example Use Cases

### Use Case 1: Review a Single CRL

**Streamlit UI:**
1. Launch app: `streamlit run streamlit_review_app.py`
2. Load knowledge base
3. Select document
4. Click "Review Document"
5. Get comprehensive analysis

**CLI:**
```bash
python review_cli.py review ./unapproved_CRLs/CRL_BLA125745_20250115.pdf
```

### Use Case 2: Batch Review 10 Documents

**Streamlit UI:**
1. Go to "Batch Review" tab
2. Select multiple documents
3. Click "Start Batch Review"
4. View all results

**CLI:**
```bash
python review_cli.py batch ./unapproved_CRLs/ --output-dir ./reviews
```

### Use Case 3: Automate Daily Reviews

```python
from crl_review_agent import CRLReviewAgent
import glob

# Setup (once)
agent = CRLReviewAgent()
agent.load_knowledge_base("./vectorstore")
agent.create_agent()

# Review all pending documents
for doc in glob.glob("./unapproved_CRLs/*.pdf"):
    result = agent.review_document(doc)
    print(f"✅ {doc}: Review complete")
```

---

## 🎯 What the AI Reviews

For each document, the agent provides:

1. **Overall Assessment** - High-level summary
2. **Compliance Check** - Does it meet FDA standards?
3. **Strengths** - What the document does well
4. **Issues & Concerns** - Potential problems
5. **Recommendations** - Specific improvements
6. **Risk Assessment** - Likelihood of approval

All grounded in examples from 202 approved CRLs!

---

## 🔧 Common Commands

```bash
# Build knowledge base (first time)
python review_cli.py build

# Review one document
python review_cli.py review ./unapproved_CRLs/document.pdf

# Review all documents
python review_cli.py batch ./unapproved_CRLs/

# Launch UI
streamlit run streamlit_review_app.py

# Easy start
./quick_start.sh
```

---

## 🐛 Troubleshooting

### "NVIDIA_API_KEY not set"
```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

### "Knowledge base not found"
```bash
python review_cli.py build
# Wait 10 minutes for it to complete
```

### "Rate limit exceeded"
Wait 1 minute and try again (or upgrade your NVIDIA API tier)

### Something else?
Check `USAGE_GUIDE.md` for detailed troubleshooting

---

## 📁 Your Project Structure

```
nvidia-hackathon/
├── approved_CRLs/              ← 202 reference CRLs
├── unapproved_CRLs/            ← Documents to review
├── vectorstore/                ← Knowledge base (generated)
│
├── crl_review_agent.py         ← Core agent code
├── streamlit_review_app.py     ← Web UI
├── review_cli.py               ← CLI tool
├── quick_start.sh              ← Easy launcher
│
├── requirements.txt            ← Dependencies
├── README.md                   ← Main overview
├── USAGE_GUIDE.md             ← Detailed guide
├── ARCHITECTURE.md             ← Technical docs
└── GETTING_STARTED.md          ← This file
```

---

## 🎓 Learning Path

**Just Want to Use It?**
→ Follow the Quick Start above

**Want to Understand How It Works?**
→ Read `ARCHITECTURE.md`

**Want to Customize It?**
→ Read `USAGE_GUIDE.md` → Configuration section

**Want to Integrate It?**
→ Read `USAGE_GUIDE.md` → Python API section

---

## 🌟 Pro Tips

1. **First time?** Use the Streamlit UI - it's the easiest
2. **Building KB?** Do it once and reuse forever (saves time!)
3. **Batch reviews?** Use the CLI - it's faster
4. **Automating?** Use the Python API
5. **Testing?** Use Nemotron Nano 9B model (faster)
6. **Production?** Use Llama 3.1 70B model (better quality)

---

## 🎉 You're Ready!

You now have a production-ready AI document review system. Here's what to do:

1. ✅ Run `./quick_start.sh`
2. ✅ Enter your API key
3. ✅ Build the knowledge base
4. ✅ Start reviewing documents!

---

## 📞 Need Help?

1. **Quick questions?** Check `USAGE_GUIDE.md`
2. **Technical details?** Check `ARCHITECTURE.md`
3. **Errors?** Check Troubleshooting in `USAGE_GUIDE.md`
4. **NVIDIA API?** Visit [docs.nvidia.com](https://docs.nvidia.com)

---

## 🚀 Next Steps

Once you're comfortable:
- Review all your unapproved CRLs
- Set up automated daily reviews
- Integrate into your existing workflow
- Customize the review prompts
- Export results to your preferred format

---

**Happy Reviewing! 🎯**

Built with ❤️ using NVIDIA NIM, LangChain, and LangGraph

