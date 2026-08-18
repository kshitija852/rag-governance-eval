"""
End-to-end demo: real ChromaDB retrieval, real Ollama-generated answer,
real DeepEval scoring (via a local Ollama judge model) - no OpenAI key,
no test fakes.

Requires (see README for install commands):
  - Ollama installed and running, with two models pulled:
      ollama pull nomic-embed-text   # embeddings
      ollama pull llama3.2:3b        # judge + answer-generation model
  - pip install chromadb ollama

Run with:
  python examples/real_demo.py
"""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from deepeval.models import OllamaModel

from agent_rag_governance import RAGPolicy
from rag_governance_eval.adapters.agt_adapter import GovernedRAGSource
from rag_governance_eval.pipeline import EvalPipeline

JUDGE_MODEL_NAME = "llama3.2:3b"
EMBED_MODEL_NAME = "nomic-embed-text"

SAMPLE_DOCS = [
    "Our refund policy allows returns within 30 days of purchase for a full refund.",
    "Refunds are processed within 5-7 business days after we receive the returned item.",
    "Digital products and gift cards are non-refundable once delivered.",
    "Shipping costs are non-refundable unless the return is due to our error.",
]


def build_chroma_retriever():
    """Loads SAMPLE_DOCS into an in-memory Chroma collection, embedded
    locally via Ollama, and returns a tiny retriever object with the
    .invoke(query) interface our adapter expects."""
    client = chromadb.Client()
    embedding_fn = OllamaEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    collection = client.get_or_create_collection(
        name="refund_policy_docs", embedding_function=embedding_fn
    )
    if collection.count() == 0:
        collection.add(
            documents=SAMPLE_DOCS,
            ids=[f"doc-{i}" for i in range(len(SAMPLE_DOCS))],
        )

    class ChromaRetriever:
        def invoke(self, query: str, k: int = 3):
            results = collection.query(query_texts=[query], n_results=k)
            return [{"text": doc} for doc in results["documents"][0]]

    return ChromaRetriever()


def make_answer_fn():
    """Uses the local Ollama chat model to generate an answer grounded
    in the retrieved contexts. Deliberately simple prompting - this is
    a demo of the eval pipeline, not a prompt-engineering exercise."""
    import ollama

    def answer_fn(query: str, contexts) -> str:
        context_block = "\n".join(f"- {c.text}" for c in contexts)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context doesn't fully answer it, say what it does say.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        )
        response = ollama.chat(
            model=JUDGE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    return answer_fn


def main():
    retriever = build_chroma_retriever()
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

    query = "How long do I have to return something, and how long do refunds take?"

    print(f"Query: {query}\n")
    answer, contexts = source.run(query)
    print(f"Retrieved {len(contexts)} context chunks:")
    for c in contexts:
        print(f"  - {c.text}")
    print(f"\nGenerated answer:\n{answer}\n")

    print("Scoring with DeepEval (local Ollama judge)... this may take a while on CPU.")
    result = pipeline.evaluate(query=query, answer=answer, contexts=[c.text for c in contexts])

    print("\n--- Eval result ---")
    print("Scores:", result.scores.as_dict())
    print("Action:", result.action.value)
    print("Triggered by:", result.triggered_by)
    print("\nAudit record written to rag_eval_audit.jsonl")


if __name__ == "__main__":
    main()
