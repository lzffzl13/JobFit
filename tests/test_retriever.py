from app.services.retriever import chunk_text, retrieve_evidence


def test_chunk_text_splits_content():
    text = "Python FastAPI Redis RAG " * 120
    chunks = chunk_text(text, source="resume", chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].source == "resume"


def test_retrieve_evidence_returns_relevant_chunks():
    chunks = chunk_text(
        "FastAPI 后端项目，使用 Redis 缓存。\n\n机器学习课程项目，使用 PyTorch。",
        source="resume",
        chunk_size=40,
        overlap=0,
    )

    result = retrieve_evidence("FastAPI Redis 后端", chunks, top_k=1)

    assert result
    assert "FastAPI" in result[0].text
