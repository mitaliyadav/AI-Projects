"""
mcp_servers/rag_server/retriever.py
-------------------------------------
Queries the persisted ChromaDB vector store and returns the top-k most
semantically similar chunks for a given query string.

The retriever opens ChromaDB fresh on each call and releases the connection
immediately after. This avoids SQLite lock conflicts when the MCP server
subprocess and the parent process both need access to the same database.

Usage
-----
    from mcp_servers.rag_server.retriever import retrieve, db_ready

    if not db_ready():
        print("Run python main.py --setup-rag first")
    else:
        chunks = retrieve("How do I use LCEL?", k=5)
        for chunk in chunks:
            print(chunk)
"""

from __future__ import annotations

from pathlib import Path

# Import at module level so these load before stdio_server() claims stdout.
# Lazy imports inside functions would cause ChromaDB's startup messages to
# be written to the MCP JSON-RPC stream, corrupting it and causing hangs.
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ------------------------------------------------------------------
# Paths — mirror what indexer.py uses
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CHROMA_DB_PATH = _PROJECT_ROOT / "mcp_servers" / "rag_server" / "chroma_db"
_COLLECTION_NAME = "langchain_docs"
_EMBED_MODEL = "nomic-embed-text"


def db_ready() -> bool:
    """Return True if the ChromaDB has been built and contains data."""
    if not _CHROMA_DB_PATH.exists():
        return False
    return any(_CHROMA_DB_PATH.iterdir())


def retrieve(query: str, k: int = 5) -> list[str]:
    """
    Embed the query with Ollama and return the top-k matching chunks
    from the ChromaDB vector store.

    Opens and releases the ChromaDB connection on every call to avoid
    SQLite lock contention between the parent process and MCP subprocesses.

    Parameters
    ----------
    query : Natural language question or search term.
    k     : Number of chunks to return (default: 5).

    Returns
    -------
    List of strings — the text content of the most relevant chunks,
    ordered by cosine similarity (most similar first).

    Raises
    ------
    RuntimeError : If the ChromaDB has not been built yet.
    """
    if not db_ready():
        raise RuntimeError(
            "ChromaDB is not initialised. "
            "Run:  python main.py --setup-rag\n"
            f"Expected location: {_CHROMA_DB_PATH}"
        )

    embeddings = OllamaEmbeddings(model=_EMBED_MODEL)
    db = Chroma(
        persist_directory=str(_CHROMA_DB_PATH),
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )
    docs = db.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def retrieve_with_scores(query: str, k: int = 5) -> list[tuple[str, float]]:
    """
    Same as retrieve() but also returns the similarity score per chunk.

    Returns
    -------
    List of (chunk_text, score) tuples. Lower score = more similar
    (ChromaDB returns L2 distance by default).
    """
    if not db_ready():
        raise RuntimeError(
            "ChromaDB is not initialised. Run: python main.py --setup-rag"
        )

    embeddings = OllamaEmbeddings(model=_EMBED_MODEL)
    db = Chroma(
        persist_directory=str(_CHROMA_DB_PATH),
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )
    results = db.similarity_search_with_score(query, k=k)
    return [(doc.page_content, float(score)) for doc, score in results]


def get_collection_stats() -> dict:
    """
    Return basic stats about the stored collection.
    Useful for verifying the index was built correctly.
    """
    if not db_ready():
        return {"status": "not_built", "count": 0}

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(_CHROMA_DB_PATH))
        collection = client.get_collection(_COLLECTION_NAME)
        count = collection.count()
        return {
            "status": "ready",
            "count": count,
            "collection": _COLLECTION_NAME,
            "path": str(_CHROMA_DB_PATH),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
