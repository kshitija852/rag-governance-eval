"""
Adapter around Microsoft's `agent-rag-governance` package.

This is the ONLY file in this project that touches AGT's API directly.
If AGT changes its interface, this is the one place to update -
everything downstream (metrics, policy, pipeline) works against the
plain (query, answer, contexts) tuple this adapter produces, not
against AGT's classes.

Verified against agent-rag-governance==4.1.0:
    RAGPolicy(allowed_collections=..., denied_collections=..., ...)
    RAGGovernor(policy, agent_id).wrap(retriever, collection=...) -> GovernedRetriever
    GovernedRetriever.invoke(query) -> list[Document-like] (same shape as
        whatever retriever was wrapped: LangChain retrievers return
        Documents with .page_content; plain callables can return dicts
        or strings depending on what you wrap)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from agent_rag_governance import RAGGovernor, RAGPolicy, GovernedRetriever


@dataclass
class RetrievedContext:
    """Normalized shape this project works with internally."""

    text: str
    metadata: dict


class GovernedRAGSource:
    """
    Wraps an AGT-governed retriever plus an answer-generating callable so
    the eval pipeline can call one method and get back everything it
    needs: the query, the generated answer, and the (governed) retrieved
    contexts - regardless of whether the underlying retriever is a
    LangChain retriever, a LlamaIndex query engine, or a plain function.
    """

    def __init__(
        self,
        retriever: Any,
        answer_fn: Callable[[str, List[RetrievedContext]], str],
        *,
        policy: Optional[RAGPolicy] = None,
        agent_id: str = "rag-governance-eval",
        collection: str = "default",
    ) -> None:
        self.policy = policy or RAGPolicy(audit_enabled=True)
        self.governor = RAGGovernor(policy=self.policy, agent_id=agent_id)
        self.governed_retriever: GovernedRetriever = self.governor.wrap(
            retriever, collection=collection
        )
        self.answer_fn = answer_fn

    def run(self, query: str, **retriever_kwargs) -> tuple[str, List[RetrievedContext]]:
        """
        Executes one governed retrieval + generation step.

        retriever_kwargs are forwarded to the wrapped retriever's
        invoke() (e.g. k=8 to pull more chunks) - AGT's GovernedRetriever
        passes **kwargs straight through to whatever it wraps, so this
        is what makes "retry with a different retrieval strategy"
        possible without bypassing governance on the retry.

        Returns (answer, contexts). Raises whatever AGT raises
        (CollectionDeniedError, RateLimitExceededError, ContentScanError)
        if the policy blocks the call - this project does not swallow
        those, since "the retrieval was denied" is itself a governance
        outcome the caller should see, not an eval concern.
        """
        raw_docs = self.governed_retriever.invoke(query, **retriever_kwargs)
        contexts = [self._normalize(doc) for doc in raw_docs]
        answer = self.answer_fn(query, contexts)
        return answer, contexts

    @staticmethod
    def _normalize(doc: Any) -> RetrievedContext:
        # LangChain-style Document
        if hasattr(doc, "page_content"):
            return RetrievedContext(
                text=doc.page_content, metadata=getattr(doc, "metadata", {}) or {}
            )
        # LlamaIndex-style NodeWithScore (or a bare Node/TextNode) - these
        # expose .get_content() and .metadata (NodeWithScore proxies both
        # through to the wrapped node), not .page_content, so they need
        # their own branch rather than falling through to str(doc).
        if hasattr(doc, "get_content"):
            return RetrievedContext(
                text=doc.get_content(), metadata=getattr(doc, "metadata", {}) or {}
            )
        # dict-shaped result
        if isinstance(doc, dict):
            return RetrievedContext(
                text=doc.get("text") or doc.get("content") or str(doc),
                metadata={k: v for k, v in doc.items() if k not in ("text", "content")},
            )
        # plain string
        return RetrievedContext(text=str(doc), metadata={})