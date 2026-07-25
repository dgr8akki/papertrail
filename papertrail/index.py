"""Embed chunks and store/retrieve them with ChromaDB.

Uses Chroma's built-in default embedder (all-MiniLM-L6-v2 via ONNX) — same
model the plan called for, without pulling in torch. The collection name is
a content hash of the PDF, so re-uploading the same file never re-embeds.
"""

import chromadb

_client = None


def _db() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=".chroma")
    return _client


def build_index(doc_id: str, chunks: list[dict]) -> None:
    col = _db().get_or_create_collection(doc_id)
    if col.count() >= len(chunks):
        return  # doc_id is a content hash, so this document is already indexed
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        # Chroma metadata values must be scalars, so bbox rides along as CSV
        metadatas=[
            {"page": c["page"], "bbox": ",".join(map(str, c["bbox"] or ()))}
            for c in chunks
        ],
    )


def retrieve(doc_id: str, query: str, k: int = 5) -> list[dict]:
    col = _db().get_collection(doc_id)
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    out = []
    for text, meta in zip(res["documents"][0], res["metadatas"][0]):
        bbox = tuple(float(v) for v in meta["bbox"].split(",")) if meta["bbox"] else None
        out.append({"text": text, "page": meta["page"], "bbox": bbox})
    return out
