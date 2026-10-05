"""
Per-thread PDF retrieval-augmented-generation: indexing an uploaded PDF into
a FAISS store, and a tool the graph can call to search it.
"""
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import EMBEDDING_MODEL_NAME, PDF_CHUNK_OVERLAP, PDF_CHUNK_SIZE

# Keyed by str(thread_id), so each chat has its own indexed document(s).
PDF_VECTORSTORES: dict[str, FAISS] = {}


@lru_cache(maxsize=1)
def _get_embeddings():
    # Loaded once, lazily (downloads the model on first use, then caches it locally).
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def process_pdf(
    file_path: str,
    thread_id: str,
    chunk_size: int = PDF_CHUNK_SIZE,
    chunk_overlap: int = PDF_CHUNK_OVERLAP,
) -> int:
    """
    Load a PDF from disk, split it into chunks, embed them, and store the
    resulting FAISS index for this thread_id. Returns the number of chunks indexed.
    Called from the Streamlit frontend right after a file is uploaded.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No extractable text found in this PDF (it may be scanned/image-only).")

    vectorstore = FAISS.from_documents(chunks, _get_embeddings())
    PDF_VECTORSTORES[str(thread_id)] = vectorstore
    return len(chunks)


def clear_pdf(thread_id: str) -> None:
    """Remove the indexed PDF for a thread (e.g. the thread itself was deleted)."""
    PDF_VECTORSTORES.pop(str(thread_id), None)


@tool
def retrieve_pdf_content(query: str, config: RunnableConfig) -> str:
    """
    Search the PDF the user has uploaded in this conversation for content
    relevant to the query. Use this whenever the user asks a question that
    could be answered from an uploaded PDF document (e.g. "what does the
    document say about X", "summarize section 2", "according to the pdf...").
    """
    thread_id = str(config["configurable"].get("thread_id"))
    vectorstore = PDF_VECTORSTORES.get(thread_id)
    if vectorstore is None:
        return "No PDF has been uploaded in this conversation yet. Ask the user to upload one first."

    docs = vectorstore.similarity_search(query, k=4)
    if not docs:
        return "No relevant content was found in the uploaded PDF for this query."

    parts = []
    for d in docs:
        page = d.metadata.get("page")
        label = f"[page {page + 1}]" if isinstance(page, int) else ""
        parts.append(f"{label} {d.page_content}".strip())
    return "\n\n---\n\n".join(parts)
