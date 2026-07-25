import os

import pymupdf
import pytest

langchain_deps = pytest.importorskip(
    "langchain_impl.rag", reason="pip install -e '.[langchain]' to run this test"
)


def test_langchain_chain_answers(tmp_path, monkeypatch):
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    monkeypatch.chdir(tmp_path)

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "The Zephyr engine produces exactly 412 horsepower.")
    pdf_path = tmp_path / "sample.pdf"
    doc.save(pdf_path)

    chain = langchain_deps.build_chain(str(pdf_path))
    result = chain.invoke("How much horsepower does the Zephyr engine produce?")
    assert "412" in result
