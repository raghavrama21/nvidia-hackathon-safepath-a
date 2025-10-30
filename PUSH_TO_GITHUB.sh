#!/bin/bash
# SafePath AI - Push to GitHub
# Repository: https://github.com/raghavrama21/nvidia-hackathon-safepath-a

echo "🚀 Pushing SafePath AI to GitHub..."
echo ""

# Initialize git if needed
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
fi

# Add remote (will skip if already exists)
echo "🔗 Adding remote repository..."
git remote add origin https://github.com/raghavrama21/nvidia-hackathon-safepath-a.git 2>/dev/null || echo "   Remote already exists"

# Check what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short

# Add all files
echo ""
echo "➕ Adding files..."
git add .

# Show status
echo ""
echo "✅ Staged files:"
git status --short

# Commit
echo ""
echo "💾 Creating commit..."
git commit -m "🧠 SafePath AI - Autonomous Regulatory Risk Intelligence Agent

Features:
- Multi-step autonomous reasoning (6-step workflow)
- Retrieval + Reranking with NVIDIA Nemotron
- Domain-specific risk scoring (CMC, Clinical, Nonclinical, etc.)
- Automated action generation (checklists, JIRA tickets)
- Cost/delay projections with ROI calculations
- Advanced Streamlit dashboard
- Built on 202 approved FDA CRLs

Tech Stack:
- NVIDIA NIM (Nemotron, NV-EmbedQA, NV-RerankQA)
- LangChain + LangGraph
- FAISS vector database
- Streamlit UI

Demonstrates:
- Autonomous AI reasoning
- Multi-step agentic workflows
- Advanced RAG with reranking
- Real-world pharma applicability"

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "✅ Done! View your repository at:"
echo "   https://github.com/raghavrama21/nvidia-hackathon-safepath-a"
