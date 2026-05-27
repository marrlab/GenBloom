"""
Linear probing evaluation module for WSI DINOv2 training.

This module provides tools for evaluating self-supervised WSI bag embeddings
using linear probing (logistic regression and KNN classifiers).

Imports are deferred to avoid numba/umap conflicts at module load time.
"""

# Lazy imports to avoid numba conflict with dinov2.distributed
def __getattr__(name):
    if name == "WSILinearProbeEvaluator":
        from .eval_linear_probe import WSILinearProbeEvaluator
        return WSILinearProbeEvaluator
    elif name == "run_linear_probe_evaluation":
        from .eval_linear_probe import run_linear_probe_evaluation
        return run_linear_probe_evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'WSILinearProbeEvaluator',
    'run_linear_probe_evaluation',
]
