"""
Tests for the PDF RAG tool. Uses a fake embeddings model instead of the real
HuggingFace one so tests run fast and offline (no model download).
"""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.backend.tools import rag


class _FakeLoader:
    def __init__(self, docs):
        self._docs = docs

    def load(self):
        return self._docs


def _cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(rag, "_get_embeddings", lambda: DeterministicFakeEmbedding(size=16))
    yield
    rag.PDF_VECTORSTORES.clear()


def _index(monkeypatch, thread_id, text="LangGraph makes agentic workflows easy."):
    docs = [Document(page_content=text, metadata={"page": 0})]
    monkeypatch.setattr(rag, "PyPDFLoader", lambda path: _FakeLoader(docs))
    return rag.process_pdf("fake.pdf", thread_id=thread_id)


def test_process_pdf_indexes_chunks(monkeypatch):
    assert _index(monkeypatch, "thread-1") == 1
    assert "thread-1" in rag.PDF_VECTORSTORES


def test_process_pdf_raises_on_empty_pdf(monkeypatch):
    monkeypatch.setattr(rag, "PyPDFLoader", lambda path: _FakeLoader([]))
    with pytest.raises(ValueError):
        rag.process_pdf("fake.pdf", thread_id="thread-2")


def test_retrieve_without_upload_returns_helpful_message():
    # NOTE: `config` is passed as the invoke() config (LangChain injects it into the tool),
    # not as part of the input dict.
    result = rag.retrieve_pdf_content.invoke({"query": "anything"}, config=_cfg("no-such-thread"))
    assert "No PDF has been uploaded" in result


def test_retrieve_finds_indexed_text(monkeypatch):
    _index(monkeypatch, "thread-3")
    result = rag.retrieve_pdf_content.invoke({"query": "agentic workflows"}, config=_cfg("thread-3"))
    assert "LangGraph" in result
    assert "[page 1]" in result


def test_pdfs_are_isolated_per_thread(monkeypatch):
    _index(monkeypatch, "thread-A", "Alpha document about apples.")
    result = rag.retrieve_pdf_content.invoke({"query": "apples"}, config=_cfg("thread-B"))
    assert "No PDF has been uploaded" in result


def test_clear_pdf_removes_index(monkeypatch):
    _index(monkeypatch, "thread-4")
    rag.clear_pdf("thread-4")
    assert "thread-4" not in rag.PDF_VECTORSTORES
