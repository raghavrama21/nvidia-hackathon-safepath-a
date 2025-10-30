#!/usr/bin/env python3
"""
Simple CLI tool for quick CRL reviews
"""

import argparse
import os
import sys
from pathlib import Path

# Fix for OpenMP on macOS (must be before other imports)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from crl_review_agent import CRLReviewAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="CRL Review Agent - AI-powered document review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review a single document
  python review_cli.py review ./unapproved_CRLs/CRL_BLA125745_20250115.pdf
  
  # Build knowledge base
  python review_cli.py build
  
  # Batch review all documents
  python review_cli.py batch ./unapproved_CRLs/
  
  # Use specific model
  python review_cli.py review document.pdf --model meta/llama-3.1-405b-instruct
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build knowledge base from approved CRLs"
    )
    build_parser.add_argument(
        "--approved-dir", default="./approved_CRLs", help="Directory with approved CRLs"
    )
    build_parser.add_argument(
        "--save-path", default="./vectorstore", help="Where to save vectorstore"
    )

    # Review command
    review_parser = subparsers.add_parser("review", help="Review a single document")
    review_parser.add_argument("document", help="Path to CRL PDF to review")
    review_parser.add_argument(
        "--model", default="nvidia/nvidia-nemotron-nano-9b-v2", help="LLM model to use"
    )
    review_parser.add_argument("--output", "-o", help="Save review to file")
    review_parser.add_argument(
        "--load-kb", default="./vectorstore", help="Path to vectorstore"
    )

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Review multiple documents")
    batch_parser.add_argument("directory", help="Directory with CRLs to review")
    batch_parser.add_argument(
        "--model", default="nvidia/nvidia-nemotron-nano-9b-v2", help="LLM model to use"
    )
    batch_parser.add_argument(
        "--output-dir", default="./reviews", help="Directory to save reviews"
    )
    batch_parser.add_argument(
        "--load-kb", default="./vectorstore", help="Path to vectorstore"
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with the agent")
    chat_parser.add_argument(
        "--model", default="nvidia/nvidia-nemotron-nano-9b-v2", help="LLM model to use"
    )
    chat_parser.add_argument(
        "--load-kb", default="./vectorstore", help="Path to vectorstore"
    )

    args = parser.parse_args()

    # Check API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ Error: NVIDIA_API_KEY environment variable not set")
        print("\nPlease set your API key:")
        print("  export NVIDIA_API_KEY='your-key-here'")
        print("\nGet your free key at: https://build.nvidia.com/")
        sys.exit(1)

    if args.command == "build":
        print("🏗️  Building knowledge base from approved CRLs...")
        print(f"📁 Source: {args.approved_dir}")
        print(f"💾 Save to: {args.save_path}")
        print("\n⏱️  This may take 5-10 minutes for 202 PDFs...\n")

        agent = CRLReviewAgent(approved_crls_dir=args.approved_dir)
        agent.build_knowledge_base()
        agent.save_knowledge_base(args.save_path)

        print("\n✅ Knowledge base built and saved successfully!")
        print(f"   Location: {args.save_path}")

    elif args.command == "review":
        print(f"🔍 Reviewing document: {args.document}")
        print(f"🤖 Using model: {args.model}")
        print(f"📚 Loading knowledge base from: {args.load_kb}\n")

        # Initialize agent
        agent = CRLReviewAgent(llm_model=args.model)

        try:
            agent.load_knowledge_base(args.load_kb)
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            print("\nTry building the knowledge base first:")
            print("  python review_cli.py build")
            sys.exit(1)

        agent.create_agent()

        # Review document
        print("🤔 Agent is analyzing...\n")
        result = agent.review_document(args.document)

        # Display results
        print("\n" + "=" * 80)
        print(f"📊 REVIEW RESULTS")
        print("=" * 80)

        # Display risk score prominently
        if result.get("risk_score") and result["risk_score"]["score"] is not None:
            risk = result["risk_score"]
            risk_emoji = (
                "🟢"
                if risk["color"] == "green"
                else "🟡" if risk["color"] == "orange" else "🔴"
            )
            print(f"\n{risk_emoji} RISK SCORE: {risk['score']}/10 - {risk['category']}")
            print("=" * 80)

        # Display deficiencies
        if result.get("deficiencies") and len(result["deficiencies"]) > 0:
            print(f"\n🚨 CRITICAL DEFICIENCIES ({len(result['deficiencies'])} found):")
            print("-" * 80)
            for i, deficiency in enumerate(result["deficiencies"], 1):
                print(f"\n{i}. {deficiency}")
            print("\n" + "=" * 80)

        print(f"\n📄 Document: {result['document_path']}")
        print(f"📑 Pages: {result['document_pages']}")
        print(f"🔄 Agent steps: {result['agent_steps']}")
        print("\n" + "-" * 80)
        print("DETAILED REVIEW:")
        print("-" * 80)
        print(result["review"])
        print("\n" + "=" * 80)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                f.write(f"Review of: {result['document_path']}\n")
                f.write(f"Pages: {result['document_pages']}\n")
                f.write(f"Agent steps: {result['agent_steps']}\n\n")

                # Include risk score
                if (
                    result.get("risk_score")
                    and result["risk_score"]["score"] is not None
                ):
                    risk = result["risk_score"]
                    f.write(f"RISK SCORE: {risk['score']}/10 - {risk['category']}\n")
                    f.write("=" * 80 + "\n\n")

                # Include deficiencies
                if result.get("deficiencies") and len(result["deficiencies"]) > 0:
                    f.write(
                        f"CRITICAL DEFICIENCIES ({len(result['deficiencies'])} found):\n"
                    )
                    f.write("-" * 80 + "\n")
                    for i, deficiency in enumerate(result["deficiencies"], 1):
                        f.write(f"\n{i}. {deficiency}\n")
                    f.write("\n" + "=" * 80 + "\n\n")

                f.write("DETAILED REVIEW:\n")
                f.write("=" * 80 + "\n")
                f.write(result["review"])
            print(f"\n💾 Review saved to: {args.output}")

    elif args.command == "batch":
        print(f"📊 Batch review mode")
        print(f"📁 Source directory: {args.directory}")
        print(f"🤖 Using model: {args.model}")
        print(f"💾 Output directory: {args.output_dir}\n")

        # Get all PDFs
        doc_dir = Path(args.directory)
        documents = sorted(list(doc_dir.glob("*.pdf")))

        if not documents:
            print(f"❌ No PDF files found in {args.directory}")
            sys.exit(1)

        print(f"Found {len(documents)} documents to review\n")

        # Initialize agent
        agent = CRLReviewAgent(llm_model=args.model)

        try:
            agent.load_knowledge_base(args.load_kb)
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            print("\nTry building the knowledge base first:")
            print("  python review_cli.py build")
            sys.exit(1)

        agent.create_agent()

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Process each document
        for i, doc_path in enumerate(documents, 1):
            print(f"[{i}/{len(documents)}] Reviewing: {doc_path.name}")

            try:
                result = agent.review_document(str(doc_path))

                # Save to file
                output_file = output_dir / f"review_{doc_path.stem}.txt"
                with open(output_file, "w") as f:
                    f.write(f"Review of: {result['document_path']}\n")
                    f.write(f"Pages: {result['document_pages']}\n")
                    f.write(f"Agent steps: {result['agent_steps']}\n\n")

                    # Include risk score
                    if (
                        result.get("risk_score")
                        and result["risk_score"]["score"] is not None
                    ):
                        risk = result["risk_score"]
                        f.write(
                            f"RISK SCORE: {risk['score']}/10 - {risk['category']}\n"
                        )
                        f.write("=" * 80 + "\n\n")

                    # Include deficiencies
                    if result.get("deficiencies") and len(result["deficiencies"]) > 0:
                        f.write(
                            f"CRITICAL DEFICIENCIES ({len(result['deficiencies'])} found):\n"
                        )
                        f.write("-" * 80 + "\n")
                        for i, deficiency in enumerate(result["deficiencies"], 1):
                            f.write(f"\n{i}. {deficiency}\n")
                        f.write("\n" + "=" * 80 + "\n\n")

                    f.write("DETAILED REVIEW:\n")
                    f.write("=" * 80 + "\n")
                    f.write(result["review"])

                print(f"  ✅ Saved to: {output_file}")

                # Print quick summary
                if (
                    result.get("risk_score")
                    and result["risk_score"]["score"] is not None
                ):
                    risk = result["risk_score"]
                    risk_emoji = (
                        "🟢"
                        if risk["color"] == "green"
                        else "🟡" if risk["color"] == "orange" else "🔴"
                    )
                    print(
                        f"      {risk_emoji} Risk: {risk['score']}/10 ({risk['category']})"
                    )
                if result.get("deficiencies"):
                    print(f"      🚨 Deficiencies: {len(result['deficiencies'])}")

            except Exception as e:
                print(f"  ❌ Error: {e}")

        print(f"\n✅ Batch review complete! Results saved to: {args.output_dir}")

    elif args.command == "chat":
        print(f"💬 Chat with SafePath Agent")
        print(f"🤖 Using model: {args.model}")
        print(f"📚 Loading knowledge base from: {args.load_kb}\n")

        # Initialize agent
        agent = CRLReviewAgent(llm_model=args.model)

        try:
            agent.load_knowledge_base(args.load_kb)
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            print("\nTry building the knowledge base first:")
            print("  python review_cli.py build")
            sys.exit(1)

        agent.create_agent()

        print("✅ Agent ready! You can now ask questions about FDA CRLs.")
        print("Type 'exit' or 'quit' to end the chat.\n")
        print("-" * 60)

        # Chat loop
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                # Create query prompt
                query_prompt = f"""
Based on the knowledge base of approved FDA CRLs, please answer this question:

{user_input}

Provide a clear, concise answer with specific examples from the approved CRLs when relevant.
Cite sources using [CRL] when referencing specific documents.
"""

                print("\n🤖 SafePath: ", end="", flush=True)

                # Get agent response
                result = agent.agent.invoke({"messages": [("user", query_prompt)]})
                response = result["messages"][-1].content

                print(response)
                print("\n" + "-" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("-" * 60)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
