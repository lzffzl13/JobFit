"""
Vector-based retrieval using BGE Embedding + ChromaDB.

Chunking uses character-level splitting with sentence-boundary detection.
Retrieval uses dense vector similarity (BGE-small-zh-v1.5) via ChromaDB.
"""

import logging
import re
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app.schemas.jobfit import Evidence

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load BGE embedding model (cached after first call)."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


def chunk_text(text: str, source: str, chunk_size: int = 700, overlap: int = 120) -> list[Evidence]:
    """Split text into overlapping chunks with sentence-boundary detection."""
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []

    chunks: list[Evidence] = []
    start = 0
    chunk_id = 1
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        window = normalized[start:end]
        if end < len(normalized):
            last_break = max(window.rfind("\n"), window.rfind("。"), window.rfind("."))
            if last_break > chunk_size * 0.55:
                end = start + last_break + 1
                window = normalized[start:end]

        chunks.append(Evidence(source=source, chunk_id=chunk_id, text=window.strip(), score=0))
        chunk_id += 1
        if end >= len(normalized):
            break
        start = max(0, end - overlap)

    return chunks


def retrieve_evidence(
    query: str,
    chunks: list[Evidence],
    top_k: int = 8,
) -> list[Evidence]:
    """Retrieve most relevant chunks using BGE embedding + ChromaDB vector search."""
    if not chunks:
        return []

    model = _get_model()
    texts = [chunk.text for chunk in chunks]

    # Encode all chunk texts into dense vectors
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    # Build an in-memory ChromaDB collection
    client = chromadb.Client()
    collection = client.create_collection(name="retrieval", metadata={"hnsw:space": "cosine"})

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": chunk.source, "chunk_id": chunk.chunk_id} for chunk in chunks],
    )

    # Encode query and search
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, len(chunks)),
    )

    # Build evidence list with similarity scores
    evidence: list[Evidence] = []
    for doc, distance, metadata in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        # ChromaDB cosine distance = 1 - cosine_similarity
        score = round(1 - distance, 4)
        evidence.append(Evidence(
            source=metadata["source"],
            chunk_id=metadata["chunk_id"],
            text=doc,
            score=max(0.0, score),
        ))

    return evidence
