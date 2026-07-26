import pymupdf
import pytest

import app


def _pdf_path(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "The Zephyr engine produces exactly 412 horsepower.")
    path = tmp_path / "sample.pdf"
    doc.save(path)
    return str(path)


def test_load_pdf_indexes_and_returns_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    doc_id, pdf_bytes, status = app.load_pdf(_pdf_path(tmp_path))
    assert doc_id and pdf_bytes
    assert "Indexed" in status


def test_load_pdf_with_no_file():
    assert app.load_pdf(None) == (None, None, "Upload a PDF to begin.")


def test_ask_without_uploaded_doc_prompts_for_upload():
    history, gallery, cleared = app.ask("What?", [], None, None)
    assert history[-1]["content"] == "Upload a PDF first."
    assert gallery == []


def test_ask_renders_verified_citation_as_gallery_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    doc_id, pdf_bytes, _ = app.load_pdf(_pdf_path(tmp_path))

    monkeypatch.setattr(
        app,
        "retrieve",
        lambda *a, **k: [{"text": "The Zephyr engine produces exactly 412 horsepower.", "page": 1, "bbox": (0, 0, 10, 10)}],
    )
    monkeypatch.setattr(
        app,
        "answer",
        lambda *a, **k: {
            "answer": "412 horsepower [1]",
            "citations": [{"n": 1, "page": 1, "bbox": (0, 0, 10, 10)}],
            "grounded": True,
        },
    )

    history, gallery, cleared = app.ask("How much horsepower?", [], doc_id, pdf_bytes)
    assert history[-1]["content"] == "412 horsepower [1]"
    assert len(gallery) == 1
    assert gallery[0][1] == "[1] page 1"
    assert cleared == ""
