"""PaperTrail — chat with a PDF, with citations you can actually verify."""

import hashlib
import io
import os

import gradio as gr
import pymupdf
from PIL import Image

from papertrail.index import build_index, retrieve
from papertrail.ingest import ingest
from papertrail.rag import answer


def load_pdf(pdf_path: str | None):
    if not pdf_path:
        return None, None, "Upload a PDF to begin."
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    doc_id = "doc-" + hashlib.sha256(pdf_bytes).hexdigest()[:16]
    build_index(doc_id, ingest(pdf_bytes))
    return doc_id, pdf_bytes, "Indexed — ask a question below."


def page_image(pdf_bytes: bytes, page_no: int, bbox) -> Image.Image:
    """Render one PDF page as an image, with the cited paragraph outlined in red."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_no - 1]
    if bbox:
        page.draw_rect(pymupdf.Rect(*bbox), color=(1, 0, 0), width=2)
    png = page.get_pixmap(dpi=110).tobytes("png")
    doc.close()
    return Image.open(io.BytesIO(png))


def ask(question: str, history: list, doc_id: str | None, pdf_bytes: bytes | None):
    if not doc_id:
        return history + [{"role": "assistant", "content": "Upload a PDF first."}], [], question

    chunks = retrieve(doc_id, question)
    result = answer(question, chunks)
    text = result["answer"]
    if not result["grounded"]:
        text += "\n\n⚠️ No citation survived verification — treat this answer with suspicion."

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text},
    ]
    gallery = [
        (page_image(pdf_bytes, c["page"], c["bbox"]), f"[{c['n']}] page {c['page']}")
        for c in result["citations"]
    ]
    return history, gallery, ""


with gr.Blocks(title="PaperTrail") as demo:
    gr.Markdown(
        "# 📄 PaperTrail\n"
        "Ask questions about a PDF — every answer cites the exact page, "
        "verified and shown highlighted on the source page below."
    )
    doc_id_state = gr.State(None)
    pdf_bytes_state = gr.State(None)

    pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
    status = gr.Markdown()
    chatbot = gr.Chatbot(label="Chat")
    question = gr.Textbox(label="Ask about the document", placeholder="What is the main contribution?")
    ask_btn = gr.Button("Ask", variant="primary")
    citations_gallery = gr.Gallery(label="Verified citations", columns=2, height=300)

    pdf_upload.change(load_pdf, inputs=pdf_upload, outputs=[doc_id_state, pdf_bytes_state, status])
    ask_btn.click(
        ask,
        inputs=[question, chatbot, doc_id_state, pdf_bytes_state],
        outputs=[chatbot, citations_gallery, question],
    )
    question.submit(
        ask,
        inputs=[question, chatbot, doc_id_state, pdf_bytes_state],
        outputs=[chatbot, citations_gallery, question],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
