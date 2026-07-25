import os

import pymupdf
import pytest

from papertrail.index import build_index, retrieve
from papertrail.ingest import ingest

FACT = "The Zephyr engine produces exactly 412 horsepower."


def _pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "This report covers the annual results of Acme Motors.")
    doc.new_page().insert_text((72, 72), FACT)
    return doc.tobytes()


def test_ingest_and_retrieve(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # .chroma gets created here
    chunks = ingest(_pdf())
    assert all(c["text"] for c in chunks)
    assert chunks[-1]["page"] == 2
    build_index("doc-test", chunks)
    top = retrieve("doc-test", "How much horsepower does the Zephyr engine make?", k=1)
    assert top[0]["page"] == 2
    assert "412" in top[0]["text"]


def test_generation_with_verified_citation():
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    from papertrail.rag import answer

    chunks = ingest(_pdf())
    result = answer("How much horsepower does the Zephyr engine produce?", chunks)
    assert "412" in result["answer"]
    assert result["grounded"]
    assert any(c["page"] == 2 for c in result["citations"])
