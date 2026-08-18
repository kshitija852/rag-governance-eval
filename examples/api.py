from __future__ import annotations

from contextlib import asynccontextmanager

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from deepeval.models import OllamaModel
from fastapi import FastAPI
from pydantic import BaseModel

from agent_rag_governance import RAGPolicy
from rag_governance_eval.adapters.agt_adapter import GovernedRAGSource
from rag_governance_eval.confidence import confidence_label
from rag_governance_eval.pipeline import EvalPipeline
from rag_governance_eval.retry import evaluate_with_retry

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "user_docs"
EMBED_MODEL_NAME = "nomic-embed-text"
JUDGE_MODEL_NAME = "llama3.2:3b"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: dict  # user-facing: {"label": ..., "description": ...} - safe to show directly
    eval: dict  # internal/for programmatic callers: raw scores, policy action, retry info


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

    def answer_fn(query: str, contexts) -> str:
        context_block = "\n".join(
            f"- [{c.metadata.get('source', '?')} p.{c.metadata.get('page', '?')}] {c.text}"
            for c in contexts
        )
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


def build_app(source: GovernedRAGSource | None = None, pipeline: EvalPipeline | None = None) -> FastAPI:
    """
    Factory, not a module-level singleton - this is what makes the API
    testable. Tests pass in a fake source/pipeline; real usage (the
    `app` object below) builds the real Chroma/Ollama-backed ones.
    """
    _source = source
    _pipeline = pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal _source, _pipeline
        if _source is None:
            _source = GovernedRAGSource(
                retriever=build_retriever(),
                answer_fn=make_answer_fn(),
                policy=RAGPolicy(allowed_collections=["public_docs"], audit_enabled=True),
                collection="public_docs",
            )
        if _pipeline is None:
            _pipeline = EvalPipeline(
                backend="deepeval",
                backend_kwargs={
                    "model": OllamaModel(model=JUDGE_MODEL_NAME),
                    "enable_safety_metrics": True,
                },
                audit_log_path="rag_eval_audit.jsonl",
            )
        yield

    app = FastAPI(title="rag-governance-eval API", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        outcome = evaluate_with_retry(_source, _pipeline, req.question)
        conf = confidence_label(outcome.result.scores.faithfulness)
        return QueryResponse(
            answer=outcome.answer,
            sources=[c.metadata.get("source", "?") for c in outcome.contexts],
            confidence={"label": conf.label, "description": conf.description},
            eval={
                "scores": outcome.result.scores.as_dict(),
                "action": outcome.result.action.value,
                "triggered_by": outcome.result.triggered_by,
                "retried": outcome.retried,
                "improved_on_retry": outcome.improved,
                "attempts": outcome.attempts,
            },
        )

    return app


app = build_app()