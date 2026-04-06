"""
mcp_servers/rag_server/indexer.py
----------------------------------
One-time setup: loads LangChain documentation, applies Semantic Chunking,
embeds with Ollama, and persists to ChromaDB.

Run via:  python main.py --setup-rag
          python -m mcp_servers.rag_server.indexer   (direct)

Subsequent runs are skipped automatically if the database already exists,
unless --force is passed to main.py.

Advanced RAG technique: Semantic Chunking
-----------------------------------------
Instead of splitting text at fixed token counts, SemanticChunker embeds
every sentence and places chunk boundaries where the cosine similarity
between adjacent sentences drops below a threshold. This produces chunks
that are coherent units of thought rather than arbitrary text windows,
which significantly improves retrieval precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CHROMA_DB_PATH = _PROJECT_ROOT / "mcp_servers" / "rag_server" / "chroma_db"
_COLLECTION_NAME = "langchain_docs"

# ------------------------------------------------------------------
# Curated LangChain documentation URLs
# Covers key concepts, tutorials, and API reference pages.
# ------------------------------------------------------------------
LANGCHAIN_DOC_URLS: list[str] = [
    # Core concepts
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/chat_models/",
    "https://python.langchain.com/docs/concepts/messages/",
    "https://python.langchain.com/docs/concepts/prompt_templates/",
    "https://python.langchain.com/docs/concepts/output_parsers/",
    "https://python.langchain.com/docs/concepts/tools/",
    "https://python.langchain.com/docs/concepts/agents/",
    "https://python.langchain.com/docs/concepts/rag/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/embeddings/",
    "https://python.langchain.com/docs/concepts/text_splitters/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/lcel/",
    # Tutorials
    "https://python.langchain.com/docs/tutorials/llm_chain/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/tutorials/agents/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
]


def db_exists() -> bool:
    """Return True if a non-empty ChromaDB already exists on disk."""
    chroma_dir = _CHROMA_DB_PATH
    if not chroma_dir.exists():
        return False
    # ChromaDB creates a chroma.sqlite3 file when data is stored
    return any(chroma_dir.iterdir())


def _check_ollama_model(model: str = "nomic-embed-text") -> None:
    """Verify the required Ollama embedding model is available."""
    import subprocess
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model not in result.stdout:
        raise RuntimeError(
            f"Ollama model '{model}' is not installed.\n"
            f"Run:  ollama pull {model}\n"
            f"Then re-run:  python main.py --setup-rag"
        )


def _load_documents() -> list:
    """
    Load LangChain documentation pages using WebBaseLoader.
    Returns a list of LangChain Document objects.

    LangChain docs are served by Mintlify and render main content inside
    elements with the CSS class "prose" (often "prose max-w-none" etc.).
    We use a regex SoupStrainer to match any element whose class attribute
    contains the word "prose", which correctly handles multi-class values.
    """
    from langchain_community.document_loaders import WebBaseLoader
    import bs4
    import re

    print(f"[indexer] Loading {len(LANGCHAIN_DOC_URLS)} documentation pages...")

    all_docs = []
    for i, url in enumerate(LANGCHAIN_DOC_URLS, 1):
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={
                    # Match any element with "prose" among its CSS classes.
                    # LangChain (Mintlify) uses "prose max-w-none" etc.
                    # Regex ensures partial-class matching works correctly.
                    "parse_only": bs4.SoupStrainer(
                        attrs={"class": re.compile(r"\bprose\b")}
                    ),
                },
            )
            docs = loader.load()
            # Filter out empty documents
            docs = [d for d in docs if d.page_content.strip()]
            # Attach source URL as metadata
            for doc in docs:
                doc.metadata["source"] = url
            all_docs.extend(docs)
            print(f"  [{i}/{len(LANGCHAIN_DOC_URLS)}] Loaded: {url}  ({sum(len(d.page_content) for d in docs)} chars)")
        except Exception as e:
            print(f"  [{i}/{len(LANGCHAIN_DOC_URLS)}] Warning: could not load {url}: {e}")

    print(f"[indexer] Total documents loaded: {len(all_docs)}")
    return all_docs


def _semantic_chunk(documents: list, embeddings) -> list:
    """
    Apply Semantic Chunking to split documents at meaning boundaries.

    SemanticChunker embeds sentences and finds breakpoints where cosine
    similarity drops, producing contextually coherent chunks.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    print("[indexer] Applying semantic chunking...")
    print("          (this embeds every sentence — may take a few minutes)")

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",  # split at top-N% similarity drops
        breakpoint_threshold_amount=90,          # conservative — larger, coherent chunks
    )

    chunks = chunker.split_documents(documents)
    print(f"[indexer] Produced {len(chunks)} semantic chunks")
    return chunks


def build_index(force_rebuild: bool = False) -> None:
    """
    Full indexing pipeline: load → chunk → embed → persist.

    Parameters
    ----------
    force_rebuild : If True, delete and recreate the database even if it exists.
    """
    if db_exists() and not force_rebuild:
        print("[indexer] ChromaDB already exists — skipping indexing.")
        print(f"          Location: {_CHROMA_DB_PATH}")
        print("          Use --force flag to rebuild from scratch.")
        return

    if force_rebuild and _CHROMA_DB_PATH.exists():
        import shutil
        print("[indexer] Force rebuild: deleting existing ChromaDB...")
        shutil.rmtree(_CHROMA_DB_PATH)

    # ------------------------------------------------------------------
    # Pre-flight: check Ollama model
    # ------------------------------------------------------------------
    _check_ollama_model("nomic-embed-text")

    # ------------------------------------------------------------------
    # Embeddings model (shared between chunker + storage)
    # ------------------------------------------------------------------
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # ------------------------------------------------------------------
    # 1. Load documents
    # ------------------------------------------------------------------
    documents = _load_documents()
    if not documents:
        raise RuntimeError(
            "[indexer] No documents loaded. Check your internet connection "
            "and that the LangChain docs URLs are accessible."
        )

    # ------------------------------------------------------------------
    # 2. Semantic chunk
    # ------------------------------------------------------------------
    chunks = _semantic_chunk(documents, embeddings)
    if not chunks:
        raise RuntimeError("[indexer] Semantic chunking produced no chunks.")

    # ------------------------------------------------------------------
    # 3. Embed + persist to ChromaDB
    # ------------------------------------------------------------------
    from langchain_chroma import Chroma

    print(f"[indexer] Embedding {len(chunks)} chunks and storing in ChromaDB...")
    print(f"          Destination: {_CHROMA_DB_PATH}")

    _CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(_CHROMA_DB_PATH),
        collection_name=_COLLECTION_NAME,
    )

    print("[indexer] ✔ Indexing complete.")
    print(f"          {len(chunks)} chunks stored in {_CHROMA_DB_PATH}")


# ------------------------------------------------------------------
# Allow running directly: python -m mcp_servers.rag_server.indexer
# ------------------------------------------------------------------
if __name__ == "__main__":
    force = "--force" in sys.argv
    build_index(force_rebuild=force)
