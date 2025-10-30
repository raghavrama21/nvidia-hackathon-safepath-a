#!/usr/bin/env python3
"""
Build Knowledge Base Once - Save for Fast Loading
This script builds the knowledge base from approved CRLs and saves it.
Run this once, then use Quick Load in the Streamlit app (5-10 seconds instead of 3 minutes).
"""

import os
import time
from pathlib import Path

# Fix for OpenMP on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from crl_review_agent import CRLReviewAgent


def main():
    print("=" * 70)
    print("🏗️  SafePath - One-Time Knowledge Base Builder")
    print("=" * 70)

    # Check if knowledge base already exists
    kb_path = "./vectorstore"
    if os.path.exists(kb_path):
        print(f"\n⚠️  Knowledge base already exists at: {kb_path}")
        response = input("Do you want to rebuild it? (y/N): ")
        if response.lower() not in ["y", "yes"]:
            print("✅ Using existing knowledge base. No action needed.")
            return
        print("\n🔄 Rebuilding knowledge base...")
    else:
        print("\n✨ Building new knowledge base...")

    # Check for approved CRLs
    approved_dir = Path("./approved_CRLs")
    if not approved_dir.exists():
        print(f"\n❌ Error: Approved CRLs directory not found: {approved_dir}")
        print("Please ensure approved_CRLs/ directory exists with PDF files.")
        return

    pdf_count = len(list(approved_dir.glob("*.pdf")))
    if pdf_count == 0:
        print(f"\n❌ Error: No PDF files found in {approved_dir}")
        return

    print(f"\n📚 Found {pdf_count} approved CRL PDFs")
    print("\n⏳ This will take approximately 3 minutes...")
    print("   Building this once will save you time on every future run!")
    print("-" * 70)

    # Create agent and build knowledge base
    start_time = time.time()

    try:
        print("\n1️⃣  Creating agent...")
        agent = CRLReviewAgent()

        print("2️⃣  Loading and processing PDFs...")
        print("   (This is the slow part - embedding 202 documents)")
        agent.build_knowledge_base()

        print("3️⃣  Saving knowledge base...")
        agent.save_knowledge_base(kb_path)

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print(f"✅ SUCCESS! Knowledge base built and saved in {elapsed:.1f} seconds")
        print("=" * 70)
        print(f"\n📁 Saved to: {kb_path}")
        print(
            f"💾 Size: ~{sum(f.stat().st_size for f in Path(kb_path).rglob('*') if f.is_file()) / (1024*1024):.1f} MB"
        )

        print("\n🚀 Next Steps:")
        print("   1. Run Streamlit app: streamlit run streamlit_review_app.py")
        print("   2. Click '⚡ Quick Load' button")
        print("   3. Start reviewing documents in 5-10 seconds!")

        print(
            "\n💡 Tip: If you update your approved CRLs, run this script again to rebuild."
        )

    except Exception as e:
        print(f"\n❌ Error building knowledge base: {e}")
        print("\nTroubleshooting:")
        print("  - Check that NVIDIA_API_KEY is set in .env file")
        print(
            "  - Ensure all dependencies are installed: pip install -r requirements.txt"
        )
        print("  - Verify internet connection for NVIDIA API access")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
