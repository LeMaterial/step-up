"""Thin wrapper around the vendored ReBind implementation.

Vendored at ``external/ReBIND``. Nothing is imported from the submodule until
``build_rebind()`` or ``get_collator()`` is first called. At that point its root
is put on ``sys.path`` (so the vendor's intra-package imports work) and three
runtime patches are applied:

- ``get_sigma_and_epsilon`` looks parameters up in the full UFF table
  (:mod:`step_up.models.lj_params`) instead of raising a KeyError on any atomic
  number above 36 — i.e. on most of the organometallic datasets.
- ``Encoder.forward`` / ``Decoder.forward`` add the Laplacian positional
  encoding out of place, so the model can be trained in fp32.
- ``REBIND.forward`` clamps predicted distances in the LJ block and scrubs
  non-finite values from the rewired adjacencies.

This is the explicit "hybrid: vendor for now, refactor later" handoff. The next
iteration will copy the model code into ``step_up.models.rebind`` natively and
delete the sys.path hack.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .common import (
    N_GLOBAL_FEATURES,
    GlobalConditionCollator,
    add_lap_out_of_place,
    apply_global_conditioning,
    global_condition_mlp,
)
from .lj_params import MAX_NODE_TYPE, UFF_LJ_PARAMETERS

_REBIND_ROOT = Path(__file__).resolve().parents[3] / "external" / "ReBIND"
# Concrete-file probe: a fresh git checkout without `--recursive` leaves
# `external/ReBIND/` as an empty directory, so `exists()` on the root passes
# but later imports fail with a confusing ModuleNotFoundError. Check for an
# actual vendored file so the error message is actionable.
_REBIND_PROBE = _REBIND_ROOT / "models" / "rebind" / "modeling_rebind.py"


def _ensure_rebind_on_path() -> None:
    """Verify the submodule is initialized and put its root on ``sys.path``.

    Raises ``FileNotFoundError`` with an actionable message if the submodule
    files aren't present. Idempotent.
    """
    if not _REBIND_PROBE.exists():
        raise FileNotFoundError(
            f"ReBind submodule files missing at {_REBIND_PROBE}. "
            "Run: git submodule update --init --recursive"
        )
    if str(_REBIND_ROOT) not in sys.path:
        sys.path.insert(0, str(_REBIND_ROOT))


# Cached references to the vendored symbols, populated by ``_load_rebind()``.
# Private so that importing them directly fails instead of silently yielding
# ``None`` before the submodule is loaded; use ``build_rebind`` / ``get_collator``.
_REBIND: Any = None
_Collator: Any = None
_REBINDConfig: Any = None
_rebind_utils: Any = None
_rebind_collating: Any = None
_rebind_modeling: Any = None

# Upstream ``forward`` methods replaced by the patches below, keyed by class, so
# tests can check the patched model against upstream.
_UPSTREAM_FORWARDS: dict[type, Any] = {}

_CONDITIONED_REBIND: Any = None

__all__ = ["build_collator", "build_rebind", "get_collator"]


def _load_rebind() -> None:
    """Import the vendored ReBind modules and apply the runtime patches.

    Deferred until first use so that importing ``step_up.models.rebind`` from a
    fresh checkout (without ``--recursive``) doesn't fail at collection time.
    Idempotent.
    """
    global _REBIND, _Collator, _REBINDConfig
    global _rebind_utils, _rebind_collating, _rebind_modeling
    if _REBIND is not None:
        return
    _ensure_rebind_on_path()
    # Imports are deferred so the vendored sys.path entry exists first.
    from models import REBIND, Collator, REBINDConfig
    from models.modules import utils
    from models.rebind import collating_rebind, modeling_rebind

    _REBIND = REBIND
    _Collator = Collator
    _REBINDConfig = REBINDConfig
    _rebind_utils = utils
    _rebind_collating = collating_rebind
    _rebind_modeling = modeling_rebind

    # Apply the three runtime patches we need (defined further down). They
    # depend on the vendored modules above so we can only call them now.
    patch_lj_parameters()
    patch_inplace_lap_addition()
    patch_rebind_forward()


# ---------------------------------------------------------------------------
# LJ-parameter extension for organometallics.
# ---------------------------------------------------------------------------
# ReBind's `get_sigma_and_epsilon` hardcodes LJ parameters for atomic-number
# indices 0..35 (i.e., Z=1..36, H through Kr), and KeyErrors on anything heavier.
# We swap in the full UFF table, which agrees with theirs exactly over Z=1..36 —
# so QM9 is untouched — and covers the 4d/5d metals, the lanthanides and iodine
# that the organometallic sets are made of.


def patch_lj_parameters() -> None:
    """Replace ``get_sigma_and_epsilon`` with one that covers the whole table.

    Idempotent: calling twice has no effect beyond the first.
    """
    if getattr(_rebind_utils, "_step_up_patched", False):
        return

    def patched_get_sigma_and_epsilon(
        mol_data: Any, drugs: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `drugs` is upstream's switch for a differently keyed table; step-up
        # never sets it, and `node_type` is always Z - 1 here.
        del drugs
        eps_list: list[float] = []
        sig_list: list[float] = []
        for atom_id in mol_data.node_type:
            idx = int(atom_id.item())
            params = UFF_LJ_PARAMETERS.get(idx)
            if params is None:
                raise ValueError(
                    f"No LJ parameters for node_type {idx} (Z={idx + 1}); the UFF table "
                    f"stops at node_type {MAX_NODE_TYPE}. A larger value means the "
                    "featurizer emitted a bad atomic number."
                )
            sigma, epsilon = params
            eps_list.append(epsilon)
            sig_list.append(sigma)
        return torch.tensor(eps_list), torch.tensor(sig_list)

    _rebind_utils.get_sigma_and_epsilon = patched_get_sigma_and_epsilon
    # `from ..modules.utils import get_sigma_and_epsilon` binds the symbol
    # directly in the collator module — re-rebind it there too.
    _rebind_collating.get_sigma_and_epsilon = patched_get_sigma_and_epsilon
    _rebind_utils._step_up_patched = True


# ---------------------------------------------------------------------------
# Molecule-level (charge / spin) conditioning
# ---------------------------------------------------------------------------
# Charge and spin enter as two scalars — the formal charge in electrons and the
# number of unpaired electrons (spin multiplicity - 1) — projected by a small MLP
# and added to every atom's embedding. Scalars rather than one-hot categories
# because BOSTMC's low-spin charges span -8..+8 with single-structure tails that
# a category would never learn, and because a scalar lets you ask the trained
# model for a charge/spin combination that never appeared in training. The MLP's
# last layer starts at zero, so a freshly built conditioned model behaves exactly
# like the unconditioned one and learns to use the conditioning from there.


def _conditioned_rebind_class() -> Any:
    """Define (once) the REBIND subclass that adds global conditioning."""
    global _CONDITIONED_REBIND
    if _CONDITIONED_REBIND is not None:
        return _CONDITIONED_REBIND

    class ConditionedREBIND(_REBIND):  # type: ignore[misc, valid-type]
        """REBIND conditioned on molecule-level charge and spin."""

        def __init__(self, config: Any, n_global_features: int = N_GLOBAL_FEATURES) -> None:
            super().__init__(config)
            self.global_cond = global_condition_mlp(n_global_features, config.d_model)

        def forward(self, **inputs):
            global_features = inputs.get("global_features")
            if global_features is not None:
                inputs["global_embedding"] = self.global_cond(global_features.to(torch.float32))
            return super().forward(**inputs)

    _CONDITIONED_REBIND = ConditionedREBIND
    return ConditionedREBIND


def _patched_encoder_forward(self, **inputs):
    """Out-of-place equivalent of ``Encoder.forward`` from vendored ReBind.

    Upstream adds the Laplacian positional encoding into ``node_embedding`` with
    an in-place slice assignment. In the decoder, that write modifies the encoder
    output after ``conformer_head`` has saved it for backward, so fp32 training
    fails with autograd's "modified by an inplace operation" error. (Upstream
    trains under fp16 autocast, where the error doesn't trigger.) The encoder's
    own write is harmless, but both blocks use a zero-padded add for symmetry.
    """
    node_attr = inputs.get("node_attr")
    node_embedding = self.node_embedding(node_attr)
    lap = inputs.get("lap_eigenvectors")
    node_embedding = add_lap_out_of_place(node_embedding, lap)
    node_embedding = apply_global_conditioning(node_embedding, inputs)
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
    node_embedding = add_lap_out_of_place(node_embedding, lap)
    inputs["node_embedding"] = node_embedding

    attn_weight_dict: dict = {}
    for i, decoder_block in enumerate(self.decoder_blocks):
        block_out = decoder_block(**inputs)
        node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
        inputs["node_embedding"] = node_embedding
        attn_weight_dict[f"decoder_block_{i}"] = attn_weight
    return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}


def patch_inplace_lap_addition() -> None:
    """Replace ``Encoder.forward`` and ``Decoder.forward`` with autograd-safe versions."""
    if getattr(_rebind_modeling, "_step_up_inplace_patched", False):
        return
    _UPSTREAM_FORWARDS[_rebind_modeling.Encoder] = _rebind_modeling.Encoder.forward
    _UPSTREAM_FORWARDS[_rebind_modeling.Decoder] = _rebind_modeling.Decoder.forward
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
    # NOTE: The variable name ``pred_conformation`` is misleading — this is
    # ReBind's intentional design (see vendored ``modeling_rebind.py``). The
    # residual head treats ``hidden_X`` (decoder output) and ``conformer_base``
    # (this tensor) as two **hidden states** of identical shape
    # ``(B, N, d_model)``, stacks them along a new last dim, computes an
    # attention score across the two channels, and projects the weighted sum
    # back to coordinates. It is NOT the predicted coordinate tensor
    # ``conformer_cache`` (shape ``(B, N, 3)``); using ``conformer_cache`` here
    # would crash on the ``torch.stack`` shape mismatch.
    #
    # Upstream stores the encoder output here, and its decoder then adds the
    # Laplacian positional encoding to that same tensor in place, so upstream's
    # residual head actually receives ``encoder output + PE``. The out-of-place
    # decoder patch no longer mutates it, so the PE is added explicitly here.
    inputs["pred_conformation"] = add_lap_out_of_place(node_embedding, inputs["lap_eigenvectors"])
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
    _UPSTREAM_FORWARDS[_rebind_modeling.REBIND] = _rebind_modeling.REBIND.forward
    _rebind_modeling.REBIND.forward = _patched_rebind_forward
    _rebind_modeling._step_up_forward_patched = True


# NOTE: patches are no longer applied at module import. ``_load_rebind()``
# applies them on first use (i.e. when ``build_rebind()`` or a ``Collator``
# instance is requested via the lazy accessors below). This lets test
# collection succeed on a fresh checkout where the submodule isn't initialized
# yet — the missing-submodule error fires only when someone actually tries to
# build the model.


def get_collator():
    """Return the (lazily loaded) ReBind ``Collator`` class."""
    _load_rebind()
    return _Collator


def build_collator(conditioned: bool = False):
    """Collator instance for the DataLoader.

    With ``conditioned=True`` it also emits the charge / spin features that a
    model from ``build_rebind(n_global_features=...)`` expects.
    """
    base = get_collator()()
    return GlobalConditionCollator(base) if conditioned else base


def build_rebind(
    n_layers: int = 8,
    d_model: int = 512,
    d_ffn: int = 1024,
    n_head: int = 8,
    atom_vocab_size: int = 513,
    dropout: float = 0.0,
    n_global_features: int = 0,
):
    """Instantiate a REBIND model from a flat keyword-style config.

    ``n_global_features > 0`` returns the variant conditioned on molecule-level
    charge and spin; pair it with ``build_collator(conditioned=True)``.
    """
    _load_rebind()
    config = _REBINDConfig(
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
    if n_global_features:
        return _conditioned_rebind_class()(config, n_global_features=n_global_features)
    return _REBIND(config)
