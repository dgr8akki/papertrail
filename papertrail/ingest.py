"""PDF -> text chunks with page/bbox metadata, via pymupdf.

Each chunk keeps the 1-based page number and the union bounding box of the
text blocks it came from, so the UI can highlight the exact paragraph(s)
a citation points to.
"""

import pymupdf

CHUNK_SIZE = 800  # chars (~200 tokens): small enough for precise citations
OVERLAP = 100


def ingest(pdf_bytes: bytes) -> list[dict]:
    """Return chunks: {"text": str, "page": int (1-based), "bbox": (x0,y0,x1,y1) | None}."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    # ponytail: chunks never cross page boundaries — keeps every citation on one
    # page; revisit only if answers routinely straddle pages.
    for pno, page in enumerate(doc, start=1):
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        buf, rect = "", None
        for x0, y0, x1, y1, text, *_ in blocks:
            text = " ".join(text.split())
            if buf and len(buf) + len(text) > CHUNK_SIZE:
                chunks.append({"text": buf, "page": pno, "bbox": tuple(rect) if rect else None})
                buf, rect = buf[-OVERLAP:], None
            buf = (buf + " " + text).strip()
            block_rect = pymupdf.Rect(x0, y0, x1, y1)
            rect = block_rect if rect is None else rect | block_rect
        if buf:
            chunks.append({"text": buf, "page": pno, "bbox": tuple(rect) if rect else None})
    doc.close()
    return chunks
