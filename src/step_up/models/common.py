"""Pieces shared by the vendored model wrappers.

Both ReBind and GTMGC descend from the same code base, so they need the same
two fixes and the same optional conditioning:

- the Laplacian positional encoding is added in place upstream, which breaks
  autograd in fp32 (see the wrappers for the details), and
- neither model takes molecule-level state, so charge and spin are projected
  and added to every atom embedding.
"""

from __future__ import annotations

from typing import Any

import torch

# Charge (electrons) and unpaired electrons (spin multiplicity - 1).
N_GLOBAL_FEATURES = 2


def add_lap_out_of_place(node_embedding: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
    """Add ``lap`` into the leading channels of ``node_embedding`` without in-place ops."""
    d = node_embedding.shape[-1]
    lap_dim = lap.shape[-1]
    if lap_dim < d:
        lap = torch.nn.functional.pad(lap, (0, d - lap_dim))
    elif lap_dim > d:
        lap = lap[..., :d]
    return node_embedding + lap


def global_condition_mlp(n_features: int, d_model: int) -> torch.nn.Module:
    """Project molecule-level scalars to ``d_model``, starting as a no-op.

    The last layer is zero-initialised, so a freshly built conditioned model
    behaves exactly like the unconditioned one and learns to use the
    conditioning from there.
    """
    mlp = torch.nn.Sequential(
        torch.nn.Linear(n_features, d_model),
        torch.nn.SiLU(),
        torch.nn.Linear(d_model, d_model),
    )
    torch.nn.init.zeros_(mlp[-1].weight)
    torch.nn.init.zeros_(mlp[-1].bias)
    return mlp


def apply_global_conditioning(node_embedding: torch.Tensor, inputs: dict[str, Any]) -> torch.Tensor:
    """Broadcast ``inputs["global_embedding"]`` over atoms, if it is there.

    Padding positions embed to zero, so they are re-masked to stay that way.
    """
    global_embedding = inputs.get("global_embedding")
    if global_embedding is None:
        return node_embedding
    node_mask = inputs.get("node_mask")
    conditioned = node_embedding + global_embedding.unsqueeze(1)
    if node_mask is not None:
        conditioned = conditioned * node_mask.unsqueeze(-1)
    return conditioned


class GlobalConditionCollator:
    """Wrap a model's collator and add ``global_features`` of shape ``(B, 2)``.

    Column 0 is the formal charge in electrons, column 1 the number of unpaired
    electrons (spin multiplicity - 1), which is 0 for a closed-shell singlet.
    Graph dicts without those fields are treated as neutral and closed-shell.
    """

    def __init__(self, base: Any) -> None:
        self.base = base

    def __call__(self, mol_sq: Any) -> dict[str, Any]:
        batch = self.base(mol_sq)
        batch["global_features"] = torch.tensor(
            [
                [float(mol.get("charge", 0.0)), float(mol.get("spin_multiplicity", 1.0)) - 1.0]
                for mol in mol_sq
            ],
            dtype=torch.float32,
        )
        return batch
