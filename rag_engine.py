"""
rag_engine.py
─────────────────────────────────────────────────────────────────────────────
RAG (Retrieval-Augmented Generation) Engine for the AI Stock Valuation System.

Responsibilities:
  1. Load and chunk one or more PDF files (annual reports, 10-Ks, MD&As).
  2. Embed the chunks and persist them in a local ChromaDB vector store.
  3. Expose a CrewAI-compatible Tool that agents can call to retrieve
     semantically relevant passages at query time.

Dependencies (add to requirements.txt):
  crewai>=0.55.0
  crewai-tools>=0.8.0
  langchain>=0.2.0
  langchain-community>=0.2.0
  langchain-google-genai>=1.0.0   # Gemini embeddings
  langchain-chroma>=0.1.0         # ChromaDB integration
  pypdf>=4.0.0                    # Pure-python PDF parser
  chromadb>=0.5.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import List

# LangChain document loaders & splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings — Gemini text-embedding-004 (free tier, 768-dim)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Vector store
from langchain_chroma import Chroma

# CrewAI base tool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR  = "./chroma_db"          # Where ChromaDB stores its data
COLLECTION_NAME     = "financial_reports"    # Logical name inside ChromaDB
CHUNK_SIZE          = 1_000                  # Characters per chunk
CHUNK_OVERLAP       = 200                    # Overlap to preserve context
TOP_K_RESULTS       = 6                      # Passages returned per query
EMBED_BATCH_SIZE    = 80                     # Chunks per API batch (free tier: 100 req/min)
EMBED_BATCH_PAUSE   = 65                     # Seconds to pause between batches


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — PDF Ingestion Helper
# ─────────────────────────────────────────────────────────────────────────────

def load_and_chunk_pdfs(pdf_paths: List[str | Path]) -> list:
    """
    Load every PDF in *pdf_paths*, split them into overlapping chunks,
    and return a flat list of LangChain Document objects.

    Args:
        pdf_paths: One or more local file paths to PDF documents.

    Returns:
        List[Document] — chunked LangChain documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # These separators preserve paragraph / sentence boundaries first.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list = []

    for path in pdf_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("Loading PDF: %s", path.name)
        loader = PyPDFLoader(str(path))
        pages  = loader.load()                        # one Document per page

        chunks = splitter.split_documents(pages)
        logger.info("  → %d chunks extracted from %s", len(chunks), path.name)

        # Tag each chunk with its source filename for provenance tracking
        for chunk in chunks:
            chunk.metadata["source_file"] = path.name

        all_chunks.extend(chunks)

    logger.info("Total chunks across all PDFs: %d", len(all_chunks))
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Vector Store Builder
# ─────────────────────────────────────────────────────────────────────────────

class VectorStoreManager:
    """
    Manages the lifecycle of the ChromaDB vector store:
      - Build from scratch (embed + persist).
      - Load an existing persisted store (skip re-embedding on re-runs).
      - Return a retriever ready for agent queries.
    """

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
        """
        google_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not google_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY must be set (env var or passed explicitly)."
            )

        # gemini-embedding-001 is the current stable embedding model on this API key.
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_key,
        )
        self._vector_store: Chroma | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def build_from_pdfs(self, pdf_paths: List[str | Path]) -> "VectorStoreManager":
        """
        Chunk the PDFs, embed them, and persist to ChromaDB.
        Call this ONCE (or when the source files change).

        Uses batched embedding with automatic rate-limit retry to stay within
        the Gemini free-tier quota of 100 embed requests per minute.
        """
        chunks = load_and_chunk_pdfs(pdf_paths)
        total  = len(chunks)
        logger.info("Embedding %d chunks → ChromaDB at %s (batch size: %d) …",
                    total, CHROMA_PERSIST_DIR, EMBED_BATCH_SIZE)

        self._vector_store = None
        for batch_start in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[batch_start: batch_start + EMBED_BATCH_SIZE]
            batch_num = batch_start // EMBED_BATCH_SIZE + 1
            total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
            logger.info("  Embedding batch %d/%d (%d chunks) …",
                        batch_num, total_batches, len(batch))

            # Retry loop for 429 rate-limit errors
            for attempt in range(1, 6):
                try:
                    if self._vector_store is None:
                        self._vector_store = Chroma.from_documents(
                            documents=batch,
                            embedding=self._embeddings,
                            collection_name=COLLECTION_NAME,
                            persist_directory=CHROMA_PERSIST_DIR,
                        )
                    else:
                        texts    = [d.page_content for d in batch]
                        metadatas = [d.metadata    for d in batch]
                        self._vector_store.add_texts(texts=texts, metadatas=metadatas)
                    break  # success
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                        wait = 65 * attempt
                        logger.warning("  Rate limit hit (attempt %d). Waiting %ds …", attempt, wait)
                        time.sleep(wait)
                    else:
                        raise  # non-rate-limit error — propagate immediately

            # Pause between batches to respect the per-minute quota
            if batch_start + EMBED_BATCH_SIZE < total:
                logger.info("  Batch %d complete. Pausing %ds to respect rate limit …",
                            batch_num, EMBED_BATCH_PAUSE)
                time.sleep(EMBED_BATCH_PAUSE)

        logger.info("Vector store built and persisted successfully.")
        return self  # allow chaining: manager.build_from_pdfs([...]).get_retriever()

    def load_existing(self) -> "VectorStoreManager":
        """
        Load a previously persisted ChromaDB store without re-embedding.
        Raises RuntimeError if the store doesn't exist yet.
        """
        if not Path(CHROMA_PERSIST_DIR).exists():
            raise RuntimeError(
                f"No persisted store found at '{CHROMA_PERSIST_DIR}'. "
                "Call build_from_pdfs() first."
            )
        self._vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info("Loaded existing vector store from %s", CHROMA_PERSIST_DIR)
        return self

    def get_retriever(self):
        """Return a LangChain retriever (MMR for diversity of results)."""
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialised. Call build_from_pdfs() or load_existing().")

        return self._vector_store.as_retriever(
            search_type="mmr",   # Maximal Marginal Relevance — balances relevance + diversity
            search_kwargs={"k": TOP_K_RESULTS, "fetch_k": TOP_K_RESULTS * 3},
        )

    def similarity_search(self, query: str) -> str:
        """
        Direct similarity search that returns a formatted string of passages.
        This is the function wired into the CrewAI Tool below.
        """
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialised.")

        docs = self._vector_store.similarity_search(query, k=TOP_K_RESULTS)

        if not docs:
            return "No relevant passages found in the financial reports for this query."

        # Format results so the LLM can easily parse source provenance
        results: list[str] = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source_file", "unknown")
            page   = doc.metadata.get("page", "?")
            results.append(
                f"[Passage {i} | Source: {source} | Page: {page}]\n{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(results)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — CrewAI-Compatible RAG Tool
# ─────────────────────────────────────────────────────────────────────────────

class RAGToolInput(BaseModel):
    """Input schema for the FinancialReportRAGTool."""
    query: str = Field(
        ...,
        description=(
            "A specific, factual question to search for in the financial reports. "
            "Examples: 'What is the company's free cash flow for FY2023?', "
            "'Describe the company's R&D investment strategy.'"
        ),
    )


class FinancialReportRAGTool(BaseTool):
    """
    CrewAI Tool that performs semantic search over ingested financial PDFs.

    Agents invoke this tool with a natural-language question; the tool
    retrieves the most relevant passages from the vector store and returns
    them as a formatted string — grounding agent responses in source text
    and preventing hallucination.
    """

    name: str = "financial_report_search"
    description: str = (
        "Search the ingested financial reports (10-K, MD&A, Annual Reports) "
        "for specific facts, figures, or statements. Use this tool whenever "
        "you need quantitative data (revenue, margins, debt), qualitative "
        "insights (management commentary, strategic risks), or any claim that "
        "must be grounded in the official documents. Input should be a precise "
        "question, not a generic keyword."
    )
    args_schema: type[BaseModel] = RAGToolInput

    # Store reference to the VectorStoreManager instance.
    # Pydantic requires explicit type annotation for custom fields in BaseTool.
    _store_manager: VectorStoreManager

    def __init__(self, store_manager: VectorStoreManager, **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass Pydantic's immutability on private attrs
        object.__setattr__(self, "_store_manager", store_manager)

    def _run(self, query: str) -> str:
        """Execute the RAG retrieval. Called internally by CrewAI."""
        logger.info("[RAG Tool] Query: %s", query)
        result = self._store_manager.similarity_search(query)
        logger.info("[RAG Tool] Returned %d chars", len(result))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_tool(
    pdf_paths: List[str | Path],
    api_key: str | None = None,
    force_rebuild: bool = False,
) -> FinancialReportRAGTool:
    """
    One-call factory: ingest PDFs (or reuse cached store) and return a
    ready-to-use CrewAI tool.

    Args:
        pdf_paths:     Paths to the PDF financial reports.
        api_key:       Google API key (falls back to GOOGLE_API_KEY env var).
        force_rebuild: If True, re-embed even if a persisted store exists.

    Returns:
        FinancialReportRAGTool — attach this to any agent that needs RAG.
    """
    manager = VectorStoreManager(api_key=api_key)

    store_exists = Path(CHROMA_PERSIST_DIR).exists()
    if force_rebuild or not store_exists:
        manager.build_from_pdfs(pdf_paths)
    else:
        logger.info("Reusing existing vector store (pass force_rebuild=True to refresh).")
        manager.load_existing()

    return FinancialReportRAGTool(store_manager=manager)