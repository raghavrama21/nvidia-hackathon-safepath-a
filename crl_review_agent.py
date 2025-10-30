"""
CRL Review Agent - AI Agent for Document Review
Uses approved CRLs as knowledge base to review unapproved submissions
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix for langchain compatibility issues (verbose, debug, and llm_cache)
import langchain_core.globals

langchain_core.globals.set_verbose(False)
langchain_core.globals.set_debug(False)
langchain_core.globals.set_llm_cache(None)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
except ImportError:
    # Fallback if nvidia endpoints not available
    from langchain_community.chat_models import ChatOpenAI as ChatNVIDIA
    from langchain_community.embeddings import HuggingFaceEmbeddings as NVIDIAEmbeddings

    NVIDIARerank = None
from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CRLReviewAgent:
    """
    AI Agent that reviews CRL documents against approved CRL knowledge base
    """

    def __init__(
        self,
        approved_crls_dir: str = "./approved_CRLs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        llm_model: str = "nvidia/nvidia-nemotron-nano-9b-v2",
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    ):
        """
        Initialize the CRL Review Agent

        Args:
            approved_crls_dir: Directory containing approved CRL PDFs
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            llm_model: NVIDIA NIM model for reasoning
            embedding_model: NVIDIA embedding model
            rerank_model: NVIDIA reranking model
        """
        self.approved_crls_dir = Path(approved_crls_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model

        # Initialize components
        self.vectorstore = None
        self.retriever = None
        self.agent = None

    def build_knowledge_base(self) -> None:
        """
        Build vector database from approved CRLs
        """
        logger.info(f"Loading approved CRLs from {self.approved_crls_dir}")

        # Load all PDF files manually to avoid verbose issues
        pdf_files = list(Path(self.approved_crls_dir).glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        docs = []
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                logger.info(f"Loading PDF {i}/{len(pdf_files)}: {pdf_path.name}")
                loader = PyPDFLoader(str(pdf_path))
                pdf_docs = loader.load()
                docs.extend(pdf_docs)
            except Exception as e:
                logger.warning(f"Error loading {pdf_path.name}: {e}")

        logger.info(f"Loaded {len(docs)} pages from {len(pdf_files)} approved CRLs")

        # Split into chunks
        logger.info("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks")

        # Create embeddings and vector store
        logger.info("Creating embeddings and FAISS vectorstore...")
        embeddings = NVIDIAEmbeddings(model=self.embedding_model, truncate="END")
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

        # Create retriever
        # Using k=5 for most relevant chunks
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        logger.info("✅ Knowledge base built successfully!")

    def save_knowledge_base(self, save_path: str = "./vectorstore") -> None:
        """
        Save the vectorstore to disk for faster loading
        """
        if self.vectorstore is None:
            raise ValueError(
                "Knowledge base not built yet. Call build_knowledge_base() first."
            )

        self.vectorstore.save_local(save_path)
        logger.info(f"✅ Knowledge base saved to {save_path}")

    def load_knowledge_base(self, load_path: str = "./vectorstore") -> None:
        """
        Load a pre-built vectorstore from disk
        """
        logger.info(f"Loading knowledge base from {load_path}")
        embeddings = NVIDIAEmbeddings(model=self.embedding_model, truncate="END")
        self.vectorstore = FAISS.load_local(
            load_path, embeddings, allow_dangerous_deserialization=True
        )

        # Create retriever
        # Using k=5 for most relevant chunks
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        logger.info("✅ Knowledge base loaded successfully!")

    def create_agent(self) -> None:
        """
        Create the AI agent with retrieval tools
        """
        if self.retriever is None:
            raise ValueError(
                "Retriever not initialized. Build or load knowledge base first."
            )

        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever=self.retriever,
            name="approved_crl_knowledge_base",
            description=(
                "Search through approved FDA Complete Response Letters (CRLs) to find "
                "relevant information, patterns, common issues, successful resolutions, "
                "and regulatory guidance. Use this to compare against the document being reviewed."
            ),
        )

        # Initialize LLM
        llm = ChatNVIDIA(model=self.llm_model, temperature=0.1)

        # Define system prompt
        system_prompt = (
            "You are an expert FDA regulatory document reviewer specializing in Complete Response Letters (CRLs).\n\n"
            "Your role is to:\n"
            "1. Review submitted CRL documents thoroughly\n"
            "2. Compare them against approved CRLs in the knowledge base\n"
            "3. Identify potential issues, inconsistencies, or areas of concern\n"
            "4. Suggest improvements based on patterns from approved CRLs\n"
            "5. Provide actionable recommendations\n\n"
            "Guidelines:\n"
            "- Use the 'approved_crl_knowledge_base' tool to search for relevant context\n"
            "- Always cite specific examples from approved CRLs when making recommendations\n"
            "- Focus on: compliance, completeness, clarity, formatting, and regulatory alignment\n"
            "- Provide structured feedback with clear sections\n"
            "- Be specific and actionable in your recommendations\n"
            "- If you find something that matches approved patterns, highlight it as a strength\n"
            "- If you find deviations, explain why they might be problematic based on approved examples\n"
        )

        # Create the agent
        self.agent = create_react_agent(
            model=llm,
            tools=[retriever_tool],
            prompt=system_prompt,
        )

        logger.info("✅ Agent created successfully!")

    def review_document(self, document_path: str) -> Dict[str, Any]:
        """
        Review a CRL document and provide comprehensive feedback

        Args:
            document_path: Path to the CRL PDF to review

        Returns:
            Dictionary containing the review results
        """
        if self.agent is None:
            raise ValueError("Agent not created. Call create_agent() first.")

        logger.info(f"Loading document to review: {document_path}")

        # Load the document to review
        loader = PyPDFLoader(document_path)
        doc_pages = loader.load()
        doc_text = "\n\n".join([page.page_content for page in doc_pages])

        logger.info(f"Document loaded: {len(doc_pages)} pages")

        # Create comprehensive review prompt
        review_prompt = f"""
Please conduct a comprehensive review of the following CRL document and provide a structured risk assessment.

DOCUMENT TO REVIEW:
{doc_text[:15000]}  # Limiting to first 15k chars for context window

REVIEW REQUIREMENTS - Please structure your response EXACTLY as follows:

## RISK SCORE
Provide a numerical risk score from 1-10 where:
- 1-3: High risk of rejection (major deficiencies)
- 4-6: Medium risk (significant issues to address)
- 7-9: Low risk (minor improvements needed)
- 10: Minimal risk (ready for approval)

Format: **Risk Score: X/10** (followed by one-line justification)

## CRITICAL DEFICIENCIES
List specific deficiencies found in this document compared to approved CRLs:
- Use bullet points
- Be specific and actionable
- Cite examples from approved CRLs using [CRL]
- Categorize by: CMC, Clinical, Nonclinical, Labeling, etc.

## COMPLIANCE ASSESSMENT
Verify alignment with FDA CRL standards based on approved examples:
- Required sections present/missing
- Data completeness
- Format and organization
- Regulatory requirements met

## STRENGTHS
Identify what this document does well (cite approved CRL examples):
- Areas that match successful patterns
- Well-documented sections
- Strong data presentation

## RECOMMENDATIONS
Provide specific, actionable improvements prioritized by importance:
1. Critical (must fix)
2. Important (should fix)
3. Minor (nice to fix)

## APPROVAL LIKELIHOOD
Overall assessment: High/Medium/Low likelihood of approval
Explanation: [Brief explanation based on deficiencies and strengths]

Use the approved_crl_knowledge_base tool extensively to ground your analysis in real examples.
"""

        logger.info("Running agent review...")

        # Run the agent
        result = self.agent.invoke({"messages": [("user", review_prompt)]})

        # Extract the final response
        final_message = result["messages"][-1].content

        # Extract risk score from response
        risk_score = self._extract_risk_score(final_message)

        # Extract deficiencies
        deficiencies = self._extract_deficiencies(final_message)

        return {
            "document_path": document_path,
            "document_pages": len(doc_pages),
            "review": final_message,
            "risk_score": risk_score,
            "deficiencies": deficiencies,
            "agent_steps": len(result["messages"]) - 1,
        }

    def _extract_risk_score(self, review_text: str) -> dict:
        """
        Extract risk score from review text

        Args:
            review_text: The review text containing risk score

        Returns:
            Dictionary with score and category
        """
        import re

        # Try to find risk score pattern
        score_match = re.search(
            r"\*\*Risk Score:\s*(\d+)/10\*\*", review_text, re.IGNORECASE
        )
        if not score_match:
            score_match = re.search(
                r"Risk Score:\s*(\d+)/10", review_text, re.IGNORECASE
            )

        if score_match:
            score = int(score_match.group(1))

            # Categorize risk
            if score >= 7:
                category = "Low Risk"
                color = "green"
            elif score >= 4:
                category = "Medium Risk"
                color = "orange"
            else:
                category = "High Risk"
                color = "red"

            return {
                "score": score,
                "max_score": 10,
                "category": category,
                "color": color,
            }

        return {"score": None, "max_score": 10, "category": "Unknown", "color": "gray"}

    def _extract_deficiencies(self, review_text: str) -> list:
        """
        Extract deficiencies list from review text

        Args:
            review_text: The review text containing deficiencies

        Returns:
            List of deficiency strings
        """
        import re

        deficiencies = []

        # Find the CRITICAL DEFICIENCIES section
        deficiencies_match = re.search(
            r"##\s*CRITICAL DEFICIENCIES\s*\n(.*?)(?=##|\Z)",
            review_text,
            re.DOTALL | re.IGNORECASE,
        )

        if deficiencies_match:
            deficiencies_text = deficiencies_match.group(1)

            # Extract bullet points
            bullets = re.findall(
                r"[-•]\s*(.+?)(?=\n[-•]|\n\n|\Z)", deficiencies_text, re.DOTALL
            )
            deficiencies = [bullet.strip() for bullet in bullets if bullet.strip()]

        return deficiencies

    def batch_review(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Review multiple documents

        Args:
            documents: List of paths to CRL PDFs to review

        Returns:
            List of review results
        """
        results = []
        for doc_path in documents:
            try:
                result = self.review_document(doc_path)
                results.append(result)
                logger.info(f"✅ Completed review of {doc_path}")
            except Exception as e:
                logger.error(f"❌ Error reviewing {doc_path}: {e}")
                results.append({"document_path": doc_path, "error": str(e)})
        return results

    def chat(self, question: str) -> str:
        """
        Chat with the agent about CRLs and regulatory topics

        Args:
            question: Question to ask the agent

        Returns:
            Agent's response
        """
        if self.agent is None:
            raise ValueError("Agent not created. Call create_agent() first.")

        logger.info(f"Chat query: {question}")

        # Create query prompt
        query_prompt = f"""
Based on the knowledge base of approved FDA CRLs, please answer this question:

{question}

Provide a clear, concise answer with specific examples from the approved CRLs when relevant.
Cite sources using [CRL] when referencing specific documents.
"""

        # Run the agent
        result = self.agent.invoke({"messages": [("user", query_prompt)]})

        # Extract the final response
        response = result["messages"][-1].content

        return response


def main():
    """
    Example usage of the CRL Review Agent
    """
    # Check for NVIDIA API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("⚠️  Please set your NVIDIA_API_KEY environment variable")
        print("Get your key from: https://build.nvidia.com/")
        return

    # Initialize agent
    print("🤖 Initializing CRL Review Agent...")
    agent = CRLReviewAgent(
        approved_crls_dir="./approved_CRLs",
    )

    # Build or load knowledge base
    try:
        print("\n📚 Loading existing knowledge base...")
        agent.load_knowledge_base("./vectorstore")
    except:
        print("\n📚 Building knowledge base from approved CRLs...")
        print("⏱️  This may take several minutes for 202 PDFs...")
        agent.build_knowledge_base()
        agent.save_knowledge_base("./vectorstore")

    # Create the agent
    print("\n🧠 Creating AI agent...")
    agent.create_agent()

    # Review an example unapproved CRL
    print("\n📄 Reviewing example document...")
    unapproved_crl = "./unapproved_CRLs/CRL_BLA125745_20250115.pdf"

    if Path(unapproved_crl).exists():
        result = agent.review_document(unapproved_crl)

        print("\n" + "=" * 80)
        print(f"📊 REVIEW RESULTS FOR: {result['document_path']}")
        print("=" * 80)
        print(f"\n{result['review']}")
        print("\n" + "=" * 80)
        print(f"Agent reasoning steps: {result['agent_steps']}")
    else:
        print(f"⚠️  Example file not found: {unapproved_crl}")


if __name__ == "__main__":
    main()
