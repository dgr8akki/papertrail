"""Generation with verified citations.

The model gets numbered excerpts and must cite by number. We then verify
each citation programmatically: a cited chunk must actually share content
words with the answer, otherwise the citation is dropped. A citation the
code checked beats a citation the model claimed.
"""

import os
import re

from openai import OpenAI

MODEL = os.environ.get("PAPERTRAIL_MODEL", "llama-3.3-70b-versatile")

SYSTEM = """\
You answer questions about a document using ONLY the numbered excerpts provided.
After every factual claim, cite the excerpt(s) it came from, like [1] or [2][3].
Quote or closely paraphrase the excerpts; do not add outside knowledge.
If the excerpts do not contain the answer, reply exactly: Not found in the document."""


def _client() -> OpenAI:
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )


def _verified(answer_text: str, chunk: dict) -> bool:
    # ponytail: 3-shared-content-words heuristic; upgrade to sentence-level
    # entailment checking if it starts passing bogus citations.
    chunk_words = set(re.findall(r"[a-z]{5,}", chunk["text"].lower()))
    answer_words = set(re.findall(r"[a-z]{5,}", answer_text.lower()))
    return len(chunk_words & answer_words) >= 3


def answer(question: str, chunks: list[dict]) -> dict:
    """Return {"answer": str, "citations": [{"n", "text", "page", "bbox"}], "grounded": bool}."""
    context = "\n\n".join(
        f"[{i}] (page {c['page']}) {c['text']}" for i, c in enumerate(chunks, 1)
    )
    resp = _client().chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    text = resp.choices[0].message.content
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)}
    citations = [
        {"n": n, **chunks[n - 1]} for n in sorted(cited) if _verified(text, chunks[n - 1])
    ]
    return {"answer": text, "citations": citations, "grounded": bool(citations)}
