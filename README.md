# rag-governance-eval

An evaluation and quality-gating layer for RAG pipelines governed by
Microsoft's [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)
(specifically its `agent-rag-governance` package).

This is an **independent companion project**, not a fork of or a
contribution to AGT. It depends on `agent-rag-governance` as a normal
PyPI package and adds a layer on top of it:

```
RAG Application
      |
agent-rag-governance   (access control, rate limiting, content scanning, audit)
      |
rag-governance-eval     <- THIS PROJECT
      |                    faithfulness, hallucination, answer relevancy,
      |                    context precision/recall, toxicity, bias, and
      |                    PII leakage - scored via DeepEval or Ragas
      v
Policy decision (pass / flag / block) + automatic retry + audit trail
```

## Why this exists

AGT's `agent-rag-governance` governs *whether an agent is allowed to
retrieve from a given collection* (access control, rate limiting,
content scanning on retrieved documents). It has no visibility into
what the LLM actually generates afterward. This project closes that
gap: it scores the *generated answer* - not just the retrieved
context - and turns low scores into real behavior, not just a logged
number.

## What it actually does (not just measures)

Earlier versions of this project only logged scores; nothing acted on
them. That's no longer true:

- **Automatic retry with expanded retrieval.** If an answer is flagged
  or blocked, the pipeline automatically re-retrieves with more
  context (a larger `k`) and re-generates, before anything is returned
  to the caller. If the retry doesn't actually score better, the
  original answer is kept - more context isn't always better, and this
  project measures that rather than assuming it. A retry that fails
  outright (e.g. a judge-model timeout) falls back to the original,
  already-valid answer rather than losing the request.
- **User-facing confidence labels**, separate from the internal
  scores. An end user never sees "faithfulness: 0.83" - they see a
  plain label (`well-grounded` / `moderate confidence` / `low
  confidence`) they can actually act on.
- **A structured, queryable audit trail** (JSONL), correlated with
  AGT's own audit log by design, with query text hashed (never stored
  raw) for the same reason AGT hashes its own.

## Metrics

Computed by default on every call:
- **Faithfulness** - is every claim in the answer traceable to the
  retrieved context?
- **Hallucination** - a separately-judged check for unsupported
  claims (not simply `1 - faithfulness`; DeepEval scores these
  independently, and in practice they can disagree - see *Known
  limitations* below).
- **Answer relevancy** - does the answer actually address the
  question asked, independent of whether it's grounded? (A faithful
  answer can still dodge the question - this catches that.)

Computed only if you pass `reference=` (a gold-standard answer) into
`EvalPipeline.evaluate(...)`:
- **Context precision** / **context recall** - without a reference,
  these are structurally impossible to compute honestly, so they come
  back as `None` (not `0.0` - the two mean different things, see
  *Known limitations*). See `examples/reference_qa.py` and
  `examples/evaluate_with_reference.py` for a real, working setup.

Opt-in (`enable_safety_metrics=True` - see *Cost of the safety
metrics* below):
- **Toxicity**, **bias**, **PII leakage** - checks on the *generated
  answer* itself, closing a real gap: AGT scans retrieved documents,
  never the model's output.

## Quick start

```bash
pip install rag-governance-eval[deepeval]
```

```python
from agent_rag_governance import RAGPolicy
from rag_governance_eval.adapters.agt_adapter import GovernedRAGSource
from rag_governance_eval.pipeline import EvalPipeline
from rag_governance_eval.retry import evaluate_with_retry
from rag_governance_eval.confidence import confidence_label

def answer_fn(query, contexts):
    # Replace with a real LLM call.
    return "your generated answer"

source = GovernedRAGSource(
    retriever=your_retriever,  # raw callable, LangChain, or LlamaIndex - see below
    answer_fn=answer_fn,
    policy=RAGPolicy(allowed_collections=["public_docs"], audit_enabled=True),
    collection="public_docs",
)

pipeline = EvalPipeline(backend="deepeval", audit_log_path="rag_eval_audit.jsonl")

outcome = evaluate_with_retry(source, pipeline, "What is our refund policy?")
conf = confidence_label(outcome.result.scores.faithfulness)

print(outcome.answer)                    # what the user sees
print(conf.label, conf.description)      # what the user is told about trust
print(outcome.result.scores.as_dict())   # internal-only, for logging/dashboards
```

## Retrieval framework support

`GovernedRAGSource` normalizes whatever a retriever returns - it works
identically whether you pass in:
- A raw callable/dict-returning retriever (see `examples/query_pdfs.py`)
- A LangChain retriever (`Chroma(...).as_retriever()` - see
  `examples/query_pdfs_langchain.py`; returns real `langchain_core`
  `Document` objects, verified to produce identical results to the raw
  path)
- A LlamaIndex retriever (returns `NodeWithScore`/`TextNode` objects)

None of this requires LangChain or LlamaIndex to be installed unless
you actually use them - the adapter checks for the relevant attributes
(`page_content`, `get_content`) at runtime rather than importing either
library.

## FastAPI endpoint

`examples/api.py` wraps the full pipeline (retrieval, generation,
retry, evaluation, confidence labeling) as `POST /query`. The answer is
always returned to the caller - the eval layer never withholds a
response for this general-purpose Q&A use case (see *Design
philosophy* below); the score and policy decision are included in the
response body for programmatic callers who want them, alongside a
plain-language `confidence` field safe to show end users directly.

```bash
uvicorn examples.api:app --port 8000
```

## Dashboard

`examples/dashboard.py` (Streamlit) visualizes the accumulated
`rag_eval_audit.jsonl` - pass/flag/block counts, scores over time, and
what triggers flags/blocks most often, plus a filterable raw table.

```bash
streamlit run examples/dashboard.py
```

## PDF ingestion

`examples/ingest_pdfs.py` chunks and embeds PDFs from a local `pdfs/`
folder into a **persistent** Chroma collection (survives across
runs), batched to avoid timing out local CPU-only embedding on larger
document sets. `examples/query_pdfs.py` queries that collection
end-to-end, including through the eval pipeline.

## Design philosophy: strict RAG, no general-knowledge fallback

The answer-generation prompt in every example script instructs the
model to answer **only** from retrieved context, and to say so
plainly if the context doesn't cover the question - not to fall back
on its own general knowledge. This is deliberate: a RAG system that
silently answers from outside its document set isn't really a RAG
system anymore. A refusal like *"the provided context doesn't mention
X"* is the CORRECT behavior, not a failure - and it correctly scores
high faithfulness, since faithfulness measures whether the answer
stays within what it was given, not whether it happens to know the
answer from elsewhere.

## Cost of the safety metrics

Each additional metric is a separate judge-LLM call. On a CPU-only
local judge (this project's actual tested setup - `llama3.2:3b` via
Ollama), this adds up: faithfulness + hallucination + answer_relevancy
is already 3 calls per attempt, and a flagged first attempt triggers a
full retry (another 3+ calls). Turning on `enable_safety_metrics=True`
adds 3 more calls *per attempt*. This project has hit real DeepEval
per-attempt timeouts (default 88.5s) at this load; raise
`DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE` if you enable safety
metrics on similar hardware. Safety metrics default to **off** for
this reason - answer_relevancy is on by default since it's one call
and directly useful; toxicity/bias/PII are opt-in.

## Known limitations (found through real, live testing - not hypothetical)

- **Hallucination scoring has been unreliable on a small local judge
  in this project's own testing.** Across many live runs, correct,
  well-grounded answers repeatedly scored `hallucination ≈ 0.5`
  regardless of actual quality, while `faithfulness` varied
  meaningfully and correctly across the same runs. `bias` has shown
  early signs of the same pattern (scoring a neutral, one-sentence
  factual refusal as maximally biased, twice). Treat flags triggered
  by `hallucination` or `bias` with real skepticism on small local
  judges; `faithfulness`, `answer_relevancy`, `toxicity`, and
  `pii_leakage` have been more consistent in this project's testing.
- **DeepEval's own metrics aren't consistent in score direction.**
  `ToxicityMetric`/`BiasMetric` score higher = worse; `PIILeakageMetric`
  scores higher = *safer* (verified by reading DeepEval's own scoring
  source). This project's thresholds account for this correctly as of
  the current version - but it's a real trap if you ever add another
  DeepEval metric yourself: check the actual scoring direction in
  DeepEval's source before assuming a consistent convention.
- **The Ragas backend (`metrics/ragas_backend.py`) is unit-tested but
  has never been run live** against real retrieval/generation in this
  project - only DeepEval has been live-verified end to end. As of
  `ragas==0.4.3`, importing it can fail outright due to a broken
  optional `langchain_community.chat_models.vertexai` import chain in
  current `langchain-community` releases - see the file's docstring
  for the full writeup and workarounds.
- **`reference_qa.py` ships with exactly one seed entry**, grounded
  only in text this project's own retrieval had already surfaced and
  verified - not fabricated. Expanding it with real gold answers
  requires actually reading your source documents; this project
  cannot do that reliably on your behalf.
- **PyPI version numbers for both real dependencies (`agent-rag-
  governance`, `deepeval`) have drifted silently during this
  project's development** without breaking anything this project
  touches - confirmed by direct testing, not assumed - but this is
  worth knowing if you pin versions loosely.

## Development

```bash
pip install -e ".[deepeval,dev]"
pytest -v
```

28 tests, covering: the AGT adapter (including LangChain/LlamaIndex
normalization), audit record creation and hashing, the policy decision
engine (including regression tests for the direction bugs found
above), the retry orchestration (including graceful failure handling),
confidence labeling, and the reference-answer mechanism. All fake/mock
based - no API key, no running Ollama instance, and no network access
required to run the suite.

## Relationship to Microsoft's toolkit

This project imports `agent-rag-governance` as a dependency and does
not modify, vendor, or redistribute any of AGT's source. If it proves
useful, the natural next step is proposing it as a documented
integration/example to the AGT maintainers - not merging its code into
their repository.

## License

MIT
