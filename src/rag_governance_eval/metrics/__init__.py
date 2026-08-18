from .base import MetricBackend, MetricScores, BackendUnavailableError

__all__ = ["MetricBackend", "MetricScores", "BackendUnavailableError"]


def get_backend(name: str = "deepeval", **kwargs) -> MetricBackend:
    """
    Factory so callers don't need to know the backend module paths.
    Backends are imported lazily inside here so choosing "deepeval"
    never triggers an attempted import of ragas, and vice versa.
    """
    if name == "deepeval":
        from .deepeval_backend import DeepEvalBackend

        return DeepEvalBackend(**kwargs)
    if name == "ragas":
        from .ragas_backend import RagasBackend

        return RagasBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r}. Choose 'deepeval' or 'ragas'.")
