#!/bin/bash

# Quick Start Script for CRL Review Agent
# This script helps you get started quickly

echo "🔬 CRL Review Agent - Quick Start"
echo "=================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "📦 Virtual environment not detected."
    echo "   Activating myenv..."
    source myenv/bin/activate
fi

# Check if .env file exists
if [[ -f ".env" ]]; then
    echo "✅ .env file found - API key will be loaded automatically"
else
    echo ""
    echo "⚠️  No .env file found!"
    echo ""
    echo "Please create a .env file with your API key:"
    echo "  echo 'NVIDIA_API_KEY=your-key-here' > .env"
    echo ""
    echo "Get your free key from: https://build.nvidia.com/"
    echo ""
fi

# Check if dependencies are installed
echo ""
echo "📚 Checking dependencies..."
python3 -c "import langchain" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""
sleep 2

streamlit run streamlit_review_app.py

