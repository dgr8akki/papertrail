"""Same pipeline as papertrail/, rebuilt with LangChain.

Trade-off vs. the hand-rolled version (see README "Two implementations"):
this is fewer lines, but citation verification and bbox-level highlighting
had to be bolted on top of abstractions that assume you only want the
answer text back — the retriever interface hands you page numbers, not
paragraph coordinates.
"""

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using ONLY the numbered excerpts. Cite every claim like [1]. "
            "If the excerpts don't contain the answer, say 'Not found in the document'.",
        ),
        ("user", "Excerpts:\n\n{context}\n\nQuestion: {question}"),
    ]
)


def build_chain(pdf_path: str, persist_dir: str = ".chroma_lc"):
    docs = PyMuPDFLoader(pdf_path).load()  # one Document per page; page kept in metadata
    splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma.from_documents(splits, embedding=embeddings, persist_directory=persist_dir)
    retriever = store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )

    def format_docs(docs):
        return "\n\n".join(f"[{i}] (page {d.metadata['page'] + 1}) {d.page_content}" for i, d in enumerate(docs, 1))

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
