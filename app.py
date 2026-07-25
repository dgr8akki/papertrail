"""PaperTrail — chat with a PDF, with citations you can actually verify."""

import hashlib

import pymupdf
import streamlit as st

from papertrail.index import build_index, retrieve
from papertrail.ingest import ingest
from papertrail.rag import answer

st.set_page_config(page_title="PaperTrail", page_icon="📄")
st.title("📄 PaperTrail")
st.caption("Ask questions about a PDF — every answer cites the exact page and paragraph, highlighted on the page.")

pdf = st.file_uploader("Upload a PDF", type="pdf")
if not pdf:
    st.stop()

pdf_bytes = pdf.getvalue()
doc_id = "doc-" + hashlib.sha256(pdf_bytes).hexdigest()[:16]
if st.session_state.get("doc_id") != doc_id:
    with st.spinner("Chunking and indexing…"):
        build_index(doc_id, ingest(pdf_bytes))
    st.session_state.update(doc_id=doc_id, pdf=pdf_bytes, history=[])


def page_image(page_no: int, bbox) -> bytes:
    """Render one PDF page as PNG, with the cited paragraph outlined in red."""
    doc = pymupdf.open(stream=st.session_state["pdf"], filetype="pdf")
    page = doc[page_no - 1]
    if bbox:
        page.draw_rect(pymupdf.Rect(*bbox), color=(1, 0, 0), width=2)
    return page.get_pixmap(dpi=110).tobytes("png")


def render_citations(citations):
    for c in citations:
        with st.expander(f"[{c['n']}] page {c['page']}"):
            st.image(page_image(c["page"], c["bbox"]))


for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []))

if question := st.chat_input("Ask about the document"):
    st.session_state["history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            chunks = retrieve(st.session_state["doc_id"], question)
            result = answer(question, chunks)
        st.markdown(result["answer"])
        if not result["grounded"]:
            st.warning("No citation survived verification — treat this answer with suspicion.")
        render_citations(result["citations"])
    st.session_state["history"].append(
        {"role": "assistant", "content": result["answer"], "citations": result["citations"]}
    )
