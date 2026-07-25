"""Ragas evaluation over a golden Q/A set.

Usage:
    pip install -e ".[eval]"
    curl -L -o evals/sample.pdf https://arxiv.org/pdf/1706.03762
    GROQ_API_KEY=... python evals/run_eval.py evals/sample.pdf evals/golden.jsonl

Uses the same Groq model as the app as the judge LLM. Metrics are the
LLM-judged trio that needs no extra embedding model: faithfulness,
context precision, context recall.
"""

import hashlib
import json
import os
import sys

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness

from papertrail.index import build_index, retrieve
from papertrail.ingest import ingest
from papertrail.rag import MODEL, answer

pdf_path, golden_path = sys.argv[1], sys.argv[2]

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()
doc_id = "doc-" + hashlib.sha256(pdf_bytes).hexdigest()[:16]
build_index(doc_id, ingest(pdf_bytes))

rows = []
with open(golden_path) as f:
    for line in f:
        g = json.loads(line)
        chunks = retrieve(doc_id, g["question"])
        result = answer(g["question"], chunks)
        rows.append(
            {
                "user_input": g["question"],
                "retrieved_contexts": [c["text"] for c in chunks],
                "response": result["answer"],
                "reference": g["ground_truth"],
            }
        )
        print(f"answered: {g['question']}")

judge = LangchainLLMWrapper(
    ChatOpenAI(
        model=MODEL,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )
)
scores = evaluate(
    EvaluationDataset.from_list(rows),
    metrics=[faithfulness, context_precision, context_recall],
    llm=judge,
)
print(scores)
