"""Thin wrapper around the vendored ReBind implementation.

Vendored at ``external/ReBIND``. We add that path to ``sys.path`` once on
import so the vendor's intra-package relative imports work. We also patch
``get_sigma_and_epsilon`` so that atomic numbers > 36 (i.e., second-row+
transition metals, lanthanides, etc.) don't raise a KeyError on the
organometallic datasets.

This is the explicit "hybrid: vendor for now, refactor later" handoff. The next
iteration will copy the model code into ``step_up.models.rebind`` natively and
delete the sys.path hack.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_REBIND_ROOT = Path(__file__).resolve().parents[3] / "external" / "ReBIND"
if not _REBIND_ROOT.exists():
    raise FileNotFoundError(
        f"ReBind submodule not found at {_REBIND_ROOT}. "
        "Run: git submodule update --init --recursive"
    )

# Put the vendor root on sys.path so its internal imports (`from .modules ...`,
# `from models import ...`) resolve. Idempotent.
if str(_REBIND_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBIND_ROOT))

from models import REBIND, Collator, REBINDConfig  # noqa: E402
from models.modules import utils as _rebind_utils  # noqa: E402
from models.rebind import collating_rebind as _rebind_collating  # noqa: E402
from models.rebind import modeling_rebind as _rebind_modeling  # noqa: E402

__all__ = ["REBIND", "Collator", "REBINDConfig", "build_rebind", "patch_lj_parameters"]


# ---------------------------------------------------------------------------
# LJ-parameter extension for organometallics.
# ---------------------------------------------------------------------------
# ReBind's `get_sigma_and_epsilon` hardcodes LJ parameters for atomic-number
# indices 0..35 (i.e., Z=1..36, H through Kr). For organometallics that include
# 4d, 5d, and f-block elements, we extend with a safe fallback. The values are
# order-of-magnitude reasonable (UFF-style sigma ~3.5 A, epsilon ~0.05 kcal/mol); the
# LJ rewiring's contribution is small relative to the bond-graph signal, and
# physically realistic dispersion for metals is dominated by short-range
# Pauli repulsion which the cutoff already handles.

_DEFAULT_LJ_SIGMA = 3.5
_DEFAULT_LJ_EPSILON = 0.05


def patch_lj_parameters() -> None:
    """Replace ``get_sigma_and_epsilon`` with a Z-tolerant version.

    Idempotent: calling twice has no effect beyond the first.
    """
    if getattr(_rebind_utils, "_step_up_patched", False):
        return

    original_dict = {
        i: {"sigma": _DEFAULT_LJ_SIGMA, "epsilon": _DEFAULT_LJ_EPSILON} for i in range(118)
    }
    # `lj_parameters` is defined inside `get_sigma_and_epsilon`. Re-read its
    # canonical entries from a one-off call by inspecting the function's
    # closure-free body: we just copy from a local clone here.
    canonical = _canonical_lj_table()
    original_dict.update(canonical)

    def patched_get_sigma_and_epsilon(
        mol_data: Any, drugs: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps_list: list[float] = []
        sig_list: list[float] = []
        for i, atom_id in enumerate(mol_data.node_type):
            if drugs:
                # We don't use the `drugs` branch in step-up. Fall through to
                # the index-based lookup, which assumes `node_type` is already
                # Z-1.
                pass
            idx = int(atom_id.item())
            params = original_dict.get(
                idx, {"sigma": _DEFAULT_LJ_SIGMA, "epsilon": _DEFAULT_LJ_EPSILON}
            )
            eps_list.append(params["epsilon"])
            sig_list.append(params["sigma"])
            del i
        return torch.tensor(eps_list), torch.tensor(sig_list)

    _rebind_utils.get_sigma_and_epsilon = patched_get_sigma_and_epsilon
    # `from ..modules.utils import get_sigma_and_epsilon` binds the symbol
    # directly in the collator module — re-rebind it there too.
    _rebind_collating.get_sigma_and_epsilon = patched_get_sigma_and_epsilon
    _rebind_utils._step_up_patched = True


def _canonical_lj_table() -> dict[int, dict[str, float]]:
    """Return ReBind's original Z=1..36 LJ parameter table (key = Z - 1)."""
    return {
        0: {"sigma": 2.886, "epsilon": 0.0440},
        1: {"sigma": 2.362, "epsilon": 0.0560},
        2: {"sigma": 2.451, "epsilon": 0.0250},
        3: {"sigma": 2.745, "epsilon": 0.0850},
        4: {"sigma": 3.637, "epsilon": 0.1800},
        5: {"sigma": 3.431, "epsilon": 0.1050},
        6: {"sigma": 3.260, "epsilon": 0.0690},
        7: {"sigma": 3.118, "epsilon": 0.0600},
        8: {"sigma": 2.996, "epsilon": 0.0500},
        9: {"sigma": 2.889, "epsilon": 0.0420},
        10: {"sigma": 2.983, "epsilon": 0.0300},
        11: {"sigma": 2.905, "epsilon": 0.1110},
        12: {"sigma": 4.008, "epsilon": 0.5050},
        13: {"sigma": 3.826, "epsilon": 0.4020},
        14: {"sigma": 3.694, "epsilon": 0.3050},
        15: {"sigma": 3.594, "epsilon": 0.2740},
        16: {"sigma": 3.516, "epsilon": 0.2270},
        17: {"sigma": 3.404, "epsilon": 0.1850},
        18: {"sigma": 3.812, "epsilon": 0.0350},
        19: {"sigma": 3.487, "epsilon": 0.2380},
        20: {"sigma": 3.316, "epsilon": 0.0190},
        21: {"sigma": 3.294, "epsilon": 0.0170},
        22: {"sigma": 3.273, "epsilon": 0.0160},
        23: {"sigma": 3.249, "epsilon": 0.0150},
        24: {"sigma": 3.210, "epsilon": 0.0130},
        25: {"sigma": 3.174, "epsilon": 0.0130},
        26: {"sigma": 3.144, "epsilon": 0.0130},
        27: {"sigma": 3.116, "epsilon": 0.0130},
        28: {"sigma": 3.083, "epsilon": 0.0050},
        29: {"sigma": 3.002, "epsilon": 0.1240},
        30: {"sigma": 4.383, "epsilon": 0.4150},
        31: {"sigma": 4.310, "epsilon": 0.3790},
        32: {"sigma": 4.280, "epsilon": 0.3090},
        33: {"sigma": 4.336, "epsilon": 0.2910},
        34: {"sigma": 4.403, "epsilon": 0.2510},
        35: {"sigma": 4.463, "epsilon": 0.2200},
    }


def _patched_encoder_forward(self, **inputs):
    """Out-of-place equivalent of ``Encoder.forward`` from vendored ReBind.

    The vendored version did an in-place slice-assignment of the Laplacian
    positional encoding into ``node_embedding``, which breaks autograd on
    modern PyTorch (>=2.6). We replace it with a zero-padded add.
    """
    node_attr = inputs.get("node_attr")
    node_embedding = self.node_embedding(node_attr)
    lap = inputs.get("lap_eigenvectors")
    node_embedding = _add_lap_out_of_place(node_embedding, lap)
    inputs["node_embedding"] = node_embedding

    attn_weight_dict: dict = {}
    for i, encoder_block in enumerate(self.encoder_blocks):
        block_out = encoder_block(**inputs)
        node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
        inputs["node_embedding"] = node_embedding
        attn_weight_dict[f"encoder_block_{i}"] = attn_weight
    return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}


def _patched_decoder_forward(self, **inputs):
    """Out-of-place equivalent of ``Decoder.forward`` from vendored ReBind."""
    node_embedding = inputs.get("node_embedding")
    lap = inputs.get("lap_eigenvectors")
    node_embedding = _add_lap_out_of_place(node_embedding, lap)
    inputs["node_embedding"] = node_embedding

    attn_weight_dict: dict = {}
    for i, decoder_block in enumerate(self.decoder_blocks):
        block_out = decoder_block(**inputs)
        node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
        inputs["node_embedding"] = node_embedding
        attn_weight_dict[f"decoder_block_{i}"] = attn_weight
    return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}


def _add_lap_out_of_place(node_embedding: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
    """Add ``lap`` into the leading channels of ``node_embedding`` without in-place ops."""
    d = node_embedding.shape[-1]
    lap_dim = lap.shape[-1]
    if lap_dim < d:
        pad = (0, d - lap_dim)
        lap = torch.nn.functional.pad(lap, pad)
    elif lap_dim > d:
        lap = lap[..., :d]
    return node_embedding + lap


def patch_inplace_lap_addition() -> None:
    """Replace ``Encoder.forward`` and ``Decoder.forward`` with autograd-safe versions."""
    if getattr(_rebind_modeling, "_step_up_inplace_patched", False):
        return
    _rebind_modeling.Encoder.forward = _patched_encoder_forward
    _rebind_modeling.Decoder.forward = _patched_decoder_forward
    _rebind_modeling._step_up_inplace_patched = True


# Minimum predicted interatomic distance, in angstroms, used to clamp the LJ
# denominator. Real bonded distances are >0.7 A; this value is small enough to
# be a no-op for any realistic prediction and large enough that ``s**6`` (which
# scales as ``sigma**6 / D**6``) cannot overflow FP32 in the LJ block.
_LJ_D_MIN = 0.1


def _patched_rebind_forward(self, **inputs):
    """Out-of-place + numerically-defensive copy of ``REBIND.forward``.

    Two changes from vendored ReBind:

    1. The predicted pairwise-distance tensor ``D_cache`` is clamped to a
       small positive minimum before being used in the LJ-force expression
       ``s = sigma / D_cache``. Without this, two non-bonded atoms whose
       predicted coordinates collide produce ``s = inf`` and downstream
       ``inf * 0`` NaNs. The diagonal mask only catches ``i == j`` collisions,
       not collisions between distinct atoms — which is exactly what kills
       full-scale QM9 training after ~100 steps. Clamp threshold is well
       below any real interatomic distance so realistic forward passes are
       unaffected.
    2. The final attraction / repulsion adjacency tensors are passed through
       ``nan_to_num`` as a belt-and-suspenders guard — if anything upstream
       still produces NaN, it gets neutralized to 0 before flowing into the
       decoder's attention bias.
    """
    conformer, node_mask = inputs.get("conformer"), inputs.get("node_mask")

    encoder_out = self.encoder(**inputs)
    node_embedding = encoder_out["node_embedding"]

    cache_out = self.conformer_head(
        conformer=conformer,
        hidden_X=node_embedding,
        padding_mask=node_mask,
        compute_loss=True,
    )
    loss_cache, conformer_cache = cache_out["loss"], cache_out["conformer_hat"]

    # CHANGE (1): clamp_min on the predicted distance matrix.
    D_cache = torch.cdist(conformer_cache, conformer_cache).detach().clamp_min(_LJ_D_MIN)
    D_M = _rebind_modeling.make_cdist_mask(node_mask)
    inputs["pred_conformation"] = node_embedding
    inputs["node_embedding"] = node_embedding

    sigma, epsilon = inputs.get("sigma"), inputs.get("epsilon")
    s = sigma / D_cache
    s6 = s**6
    LJ_force_orig = 24 * epsilon * (s6 / D_cache) * (2 * s6 - 1)
    LJ_force = torch.abs(LJ_force_orig)

    adj_mask = inputs["adjacency"].bool()
    self_mask = (
        torch.eye(D_cache.shape[1], device=D_cache.device)
        .bool()
        .unsqueeze(0)
        .expand(D_cache.shape[0], -1, -1)
    )
    LJ_force = LJ_force.masked_fill(adj_mask | self_mask, 0)
    LJ_force = LJ_force.masked_fill(~D_M.bool(), 0)

    num_retain_edges = inputs.get("num_near_edges")
    ret_val = _rebind_modeling.retain_top_k(LJ_force, num_retain_edges, descending=True)
    nonzeros = ret_val.nonzero(as_tuple=False)
    ret_val[nonzeros[:, 0], nonzeros[:, 1], nonzeros[:, 2]] = 1

    LJ_nonzero_mask = LJ_force > 0
    LJ_above_zero_mask = LJ_force_orig > 0
    LJ_below_zero_mask = LJ_force_orig < 0

    repulsive = LJ_nonzero_mask.float() * LJ_above_zero_mask.float() * ret_val * -1
    attractive = LJ_nonzero_mask.float() * LJ_below_zero_mask.float() * ret_val * 1
    # CHANGE (2): scrub any residual NaN/inf out of the attention biases.
    inputs["attraction_adjacency"] = torch.nan_to_num(attractive, nan=0.0, posinf=0.0, neginf=0.0)
    inputs["repulsion_adjacency"] = torch.nan_to_num(repulsive, nan=0.0, posinf=0.0, neginf=0.0)

    decoder_out = self.decoder(**inputs)
    node_embedding = decoder_out["node_embedding"]

    outputs = self.residual_head(
        conformer=conformer,
        hidden_X=node_embedding,
        padding_mask=node_mask,
        compute_loss=True,
        conformer_base=inputs["pred_conformation"],
    )

    return _rebind_modeling.ConformerPredictionOutput(
        loss=(outputs["loss"] + loss_cache) / 2,
        cdist_mae=outputs["cdist_mae"],
        cdist_mse=outputs["cdist_mse"],
        coord_rmsd=outputs["coord_rmsd"],
        conformer=outputs["conformer"],
        conformer_hat=outputs["conformer_hat"],
    )


def patch_rebind_forward() -> None:
    """Replace ``REBIND.forward`` with the numerically-defensive version."""
    if getattr(_rebind_modeling, "_step_up_forward_patched", False):
        return
    _rebind_modeling.REBIND.forward = _patched_rebind_forward
    _rebind_modeling._step_up_forward_patched = True


# Apply patches eagerly on import.
patch_lj_parameters()
patch_inplace_lap_addition()
patch_rebind_forward()


def build_rebind(
    n_layers: int = 8,
    d_model: int = 512,
    d_ffn: int = 1024,
    n_head: int = 8,
    atom_vocab_size: int = 513,
    dropout: float = 0.0,
) -> REBIND:
    """Instantiate a REBIND model from a flat keyword-style config."""
    config = REBINDConfig(
        n_encode_layers=n_layers,
        n_decode_layers=n_layers,
        embed_style="atom_type_ids",
        atom_vocab_size=atom_vocab_size,
        d_embed=d_model,
        pre_ln=False,
        d_q=d_model,
        d_k=d_model,
        d_v=d_model,
        d_model=d_model,
        n_head=n_head,
        qkv_bias=True,
        attn_drop=dropout,
        norm_drop=dropout,
        ffn_drop=dropout,
        dropout=dropout,
        d_ffn=d_ffn,
    )
    return REBIND(config)
