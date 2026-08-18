"""
Queries the persistent Chroma collection built by ingest_pdfs.py,
generates an answer with Ollama, and scores it with the eval pipeline -
same flow as real_demo.py, but against your real ingested PDFs instead
of the 4 hardcoded sample sentences.

Run with:
  python examples/query_pdfs.py "What is the refund window?"
"""

from __future__ import annotations

import sys

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from deepeval.models import OllamaModel

from agent_rag_governance import RAGPolicy
from rag_governance_eval.adapters.agt_adapter import GovernedRAGSource
from rag_governance_eval.pipeline import EvalPipeline

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "user_docs"
EMBED_MODEL_NAME = "nomic-embed-text"
JUDGE_MODEL_NAME = "llama3.2:3b"


def build_retriever():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = OllamaEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception:
        print(f"No collection found at {CHROMA_PATH}. Run ingest_pdfs.py first.")
        sys.exit(1)

    class ChromaRetriever:
        def invoke(self, query: str, k: int = 4):
            results = collection.query(query_texts=[query], n_results=k)
            docs, metas = results["documents"][0], results["metadatas"][0]
            return [{"text": d, **m} for d, m in zip(docs, metas)]

    return ChromaRetriever()


def make_answer_fn():
    import ollama

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
        response = ollama.chat(
            model=JUDGE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},  # deterministic: same prompt -> same
            # (or near-same) answer each run, so we can tell apart
            # "generation varies" from "judge scoring varies"
        )
        return response["message"]["content"]

    return answer_fn


def main():
    if len(sys.argv) < 2:
        print('Usage: python examples/query_pdfs.py "your question here"')
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

    # --- INTERNAL/OBSERVABILITY: scoring still runs, still gets written
    print("\n--- [internal] eval result (not shown to end user) ---")
    print("Scoring (this may take a while on CPU)...")
    result = pipeline.evaluate(query=query, answer=answer, contexts=[c.text for c in contexts])
    print("Scores:", result.scores.as_dict())
    print("Action:", result.action.value, "| Triggered by:", result.triggered_by)
    print("(audit record appended to rag_eval_audit.jsonl)")


if __name__ == "__main__":
    main()