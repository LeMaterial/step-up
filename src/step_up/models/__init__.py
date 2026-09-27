"""Model wrappers for step-up benchmark.

Each benchmark model is a thin wrapper over a vendored upstream implementation
and exposes the same two entry points, so the training loop stays model-agnostic:
``build_model`` for the network and ``build_model_collator`` for its batching.
"""

from __future__ import annotations

from typing import Any

from .common import N_GLOBAL_FEATURES

MODEL_NAMES = ("rebind", "gtmgc")


def build_model(
    name: str,
    n_layers: int,
    d_model: int,
    d_ffn: int,
    n_head: int,
    dropout: float = 0.0,
    n_global_features: int = 0,
) -> Any:
    """Build one of the benchmark models by name."""
    if name == "rebind":
        from .rebind import build_rebind

        return build_rebind(
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            n_head=n_head,
            dropout=dropout,
            n_global_features=n_global_features,
        )
    if name == "gtmgc":
        from .gtmgc import build_gtmgc

        return build_gtmgc(
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            n_head=n_head,
            dropout=dropout,
            n_global_features=n_global_features,
        )
    raise ValueError(f"Unknown model: {name!r} (expected one of {MODEL_NAMES})")


def build_model_collator(name: str, conditioned: bool = False) -> Any:
    """Build the collator that matches ``name``'s expected batch layout."""
    if name == "rebind":
        from .rebind import build_collator

        return build_collator(conditioned=conditioned)
    if name == "gtmgc":
        from .gtmgc import build_gtmgc_collator

        return build_gtmgc_collator(conditioned=conditioned)
    raise ValueError(f"Unknown model: {name!r} (expected one of {MODEL_NAMES})")


__all__ = ["MODEL_NAMES", "N_GLOBAL_FEATURES", "build_model", "build_model_collator"]
