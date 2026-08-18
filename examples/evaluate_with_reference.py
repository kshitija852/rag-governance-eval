"""
Runs a question from reference_qa.py through the real pipeline WITH a
reference answer supplied - this is what actually produces real
context_precision/context_recall numbers instead of the None you've
seen in every run so far.

Run with:
  python examples/evaluate_with_reference.py
"""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from deepeval.models import OllamaModel

from agent_rag_governance import RAGPolicy
from rag_governance_eval.adapters.agt_adapter import GovernedRAGSource
from rag_governance_eval.pipeline import EvalPipeline
from reference_qa import REFERENCE_QA

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "user_docs"
EMBED_MODEL_NAME = "nomic-embed-text"
JUDGE_MODEL_NAME = "llama3.2:3b"


def build_retriever():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = OllamaEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    class ChromaRetriever:
        def invoke(self, query: str, k: int = 4):
            results = collection.query(query_texts=[query], n_results=k)
            docs, metas = results["documents"][0], results["metadatas"][0]
            return [{"text": d, **m} for d, m in zip(docs, metas)]

    return ChromaRetriever()


def make_answer_fn():
    import ollama

    def answer_fn(query, contexts) -> str:
        context_block = "\n".join(f"- {c.text}" for c in contexts)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context doesn't answer it, say so plainly.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        )
        response = ollama.chat(
            model=JUDGE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return response["message"]["content"]

    return answer_fn


def main():
    source = GovernedRAGSource(
        retriever=build_retriever(),
        answer_fn=make_answer_fn(),
        policy=RAGPolicy(allowed_collections=["public_docs"], audit_enabled=True),
        collection="public_docs",
    )
    pipeline = EvalPipeline(
        backend="deepeval",
        backend_kwargs={"model": OllamaModel(model=JUDGE_MODEL_NAME)},
        audit_log_path="rag_eval_audit.jsonl",
    )

    for item in REFERENCE_QA:
        query, reference = item["question"], item["reference"]
        print(f"Query: {query}")
        answer, contexts = source.run(query)
        print(f"Answer: {answer}\n")

        result = pipeline.evaluate(
            query=query,
            answer=answer,
            contexts=[c.text for c in contexts],
            reference=reference,
        )
        print("Scores (now with REAL precision/recall):", result.scores.as_dict())
        print("Action:", result.action.value, "\n" + "-" * 60)


if __name__ == "__main__":
    main()