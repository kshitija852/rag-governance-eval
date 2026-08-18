"""
Same task as query_pdfs.py - retrieve from the persistent Chroma
collection, generate an answer, evaluate it - but built with LangChain
instead of raw chromadb/ollama calls, so you can compare the two
approaches directly.

Requires (on top of what query_pdfs.py needs):
  pip install langchain-chroma langchain-ollama langchain-core

Run with:
  python examples/query_pdfs_langchain.py "your question here"
"""

from __future__ import annotations

import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from deepeval.models import OllamaModel

from agent_rag_governance import RAGPolicy
from rag_governance_eval.adapters.agt_adapter import GovernedRAGSource
from rag_governance_eval.pipeline import EvalPipeline

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "user_docs"
EMBED_MODEL_NAME = "nomic-embed-text"
JUDGE_MODEL_NAME = "llama3.2:3b"


def build_retriever():
    """
    This is the entire retrieval setup with LangChain - compare this
    function to build_retriever() in query_pdfs.py. LangChain's
    .as_retriever() gives us a standard object with an .invoke(query)
    method, matching what GovernedRAGSource already expects, and its
    results come back as real langchain_core Document objects - which
    the adapter's _normalize() already handles via the page_content
    branch (no change needed there for LangChain specifically).
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def make_answer_fn():
    llm = ChatOllama(model=JUDGE_MODEL_NAME, temperature=0)

    def answer_fn(query: str, contexts) -> str:
        context_block = "\n".join(
            f"- [{c.metadata.get('source', '?')} p.{c.metadata.get('page', '?')}] {c.text}"
            for c in contexts
        )
        prompt = (
            "Answer the question using ONLY the context below, which is excerpted "
            "from real source documents. Cite the source filename when relevant. "
            "If the context doesn't answer it, say so plainly.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        )
        response = llm.invoke(prompt)
        return response.content

    return answer_fn


def main():
    if len(sys.argv) < 2:
        print('Usage: python examples/query_pdfs_langchain.py "your question here"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    retriever = build_retriever()
    answer_fn = make_answer_fn()

    source = GovernedRAGSource(
        retriever=retriever,
        answer_fn=answer_fn,
        policy=RAGPolicy(allowed_collections=["public_docs"], audit_enabled=True),
        collection="public_docs",
    )

    pipeline = EvalPipeline(
        backend="deepeval",
        backend_kwargs={"model": OllamaModel(model=JUDGE_MODEL_NAME)},
        audit_log_path="rag_eval_audit.jsonl",
    )

    print(f"Query: {query}\n")
    answer, contexts = source.run(query)
    print(f"Retrieved {len(contexts)} chunk(s):")
    for c in contexts:
        print(f"  [{c.metadata.get('source')} p.{c.metadata.get('page')}] {c.text[:100]}...")

    print("\nAnswer:\n")
    print(answer)

    print("\n--- [internal] eval result (not shown to end user) ---")
    print("Scoring (this may take a while on CPU)...")
    result = pipeline.evaluate(query=query, answer=answer, contexts=[c.text for c in contexts])
    print("Scores:", result.scores.as_dict())
    print("Action:", result.action.value, "| Triggered by:", result.triggered_by)


if __name__ == "__main__":
    main()