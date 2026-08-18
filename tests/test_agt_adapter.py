"""
These tests exercise the REAL `agent-rag-governance` package (not a
mock) - it's a pure local policy engine with no external calls, so
there's no reason to fake it. What's faked is the underlying retriever
and answer function, since those would normally be a real vector store
and LLM call.
"""

import pytest
from agent_rag_governance import RAGPolicy
from agent_rag_governance.exceptions import CollectionDeniedError

from rag_governance_eval.adapters.agt_adapter import GovernedRAGSource, RetrievedContext


class FakeDoc:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {"source": "fake"}


class FakeRetriever:
    """Mimics a LangChain retriever's .invoke() -> list[Document]."""

    def invoke(self, query: str):
        return [FakeDoc(f"context relevant to: {query}")]


def fake_answer_fn(query, contexts):
    joined = " ".join(c.text for c in contexts)
    return f"Answer for '{query}' using: {joined}"


def test_governed_source_returns_normalized_contexts():
    source = GovernedRAGSource(
        retriever=FakeRetriever(),
        answer_fn=fake_answer_fn,
        policy=RAGPolicy(audit_enabled=False),
        collection="public_docs",
    )
    answer, contexts = source.run("what is our refund policy?")

    assert "Answer for" in answer
    assert len(contexts) == 1
    assert isinstance(contexts[0], RetrievedContext)
    assert "refund policy" in contexts[0].text


def test_denied_collection_raises_agt_error():
    source = GovernedRAGSource(
        retriever=FakeRetriever(),
        answer_fn=fake_answer_fn,
        policy=RAGPolicy(denied_collections=["financial_data"], audit_enabled=False),
        collection="financial_data",
    )
    with pytest.raises(CollectionDeniedError):
        source.run("show me the numbers")

class LangChainStyleRetriever:
    def invoke(self, query: str):
        from langchain_core.documents import Document
        return [Document(page_content=f"langchain result for: {query}", metadata={"source": "lc"})]


def test_normalizes_langchain_document():
    source = GovernedRAGSource(
        retriever=LangChainStyleRetriever(),
        answer_fn=fake_answer_fn,
        policy=RAGPolicy(audit_enabled=False),
        collection="public_docs",
    )
    _, contexts = source.run("test query")
    assert contexts[0].text == "langchain result for: test query"
    assert contexts[0].metadata == {"source": "lc"}


class LlamaIndexStyleRetriever:
    def invoke(self, query: str):
        from llama_index.core.schema import NodeWithScore, TextNode
        node = TextNode(text=f"llamaindex result for: {query}", metadata={"source": "li"})
        return [NodeWithScore(node=node, score=0.95)]


def test_normalizes_llamaindex_node():
    source = GovernedRAGSource(
        retriever=LlamaIndexStyleRetriever(),
        answer_fn=fake_answer_fn,
        policy=RAGPolicy(audit_enabled=False),
        collection="public_docs",
    )
    _, contexts = source.run("test query")
    assert contexts[0].text == "llamaindex result for: test query"
    assert contexts[0].metadata == {"source": "li"}
