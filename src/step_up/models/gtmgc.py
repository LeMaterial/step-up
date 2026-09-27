"""Thin wrapper around the vendored GTMGC implementation.

Vendored at ``external/GTMGC`` (Xu et al., ICLR 2024). ReBind was built on this
code base, so the two wrappers look alike. GTMGC's package is *also* called
``models``, which would collide with ReBind's entry on ``sys.path``, so this one
is imported under a private module name instead of by path injection.

Nothing is imported until ``build_gtmgc()`` or ``get_gtmgc_collator()`` runs. One
runtime patch is applied: ``GTMGCEncoder.forward`` and ``GTMGCDecoder.forward``
add the Laplacian positional encoding with an in-place slice assignment, which
breaks autograd in fp32 exactly as ReBind's does (upstream trains under fp16
autocast). Unlike ReBind, nothing reads the tensor between the write and its use,
so the out-of-place replacement is value-identical to upstream.

Their conformer models embed atoms as Mole-BERT tokenizer ids
(``embed_style="atom_tokenized_ids"``), so batches need per-atom ``input_ids``;
see ``scripts/tokenize_molebert.py``.
"""

from __future__ import annotations

import importlib.util
import json
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

_GTMGC_ROOT = Path(__file__).resolve().parents[3] / "external" / "GTMGC"
_GTMGC_PROBE = _GTMGC_ROOT / "models" / "gtmgc" / "modeling_gtmgc.py"
_VENDOR_MODULE = "_step_up_vendor_gtmgc"

_gtmgc: Any = None
_gtmgc_modeling: Any = None
# Upstream ``forward`` methods replaced below, keyed by class, so tests can check
# the patched model against upstream.
_UPSTREAM_FORWARDS: dict[type, Any] = {}
_CONDITIONED_GTMGC: Any = None

__all__ = ["build_gtmgc", "build_gtmgc_collator", "get_gtmgc_collator"]


def _load_gtmgc() -> None:
    """Import the vendored GTMGC package and apply the runtime patch. Idempotent."""
    global _gtmgc, _gtmgc_modeling
    if _gtmgc is not None:
        return
    if not _GTMGC_PROBE.exists():
        raise FileNotFoundError(
            f"GTMGC submodule files missing at {_GTMGC_PROBE}. "
            "Run: git submodule update --init --recursive"
        )
    package = _GTMGC_ROOT / "models"
    spec = importlib.util.spec_from_file_location(
        _VENDOR_MODULE, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing so the package's relative imports resolve.
    sys.modules[_VENDOR_MODULE] = module
    spec.loader.exec_module(module)
    _gtmgc = module
    _gtmgc_modeling = sys.modules[f"{_VENDOR_MODULE}.gtmgc.modeling_gtmgc"]
    patch_inplace_lap_addition()


def _patched_encoder_forward(self, **inputs):
    """Out-of-place equivalent of ``GTMGCEncoder.forward``."""
    if self.embed_style == "atom_tokenized_ids":
        node_embedding = self.node_embedding(inputs.get("node_input_ids"))
    elif self.embed_style == "atom_type_ids":
        node_embedding = self.node_embedding(inputs.get("node_type"))
    elif self.embed_style == "ogb":
        node_embedding = self.ogb_node_embedding(inputs["node_attr"])
    else:
        raise ValueError(f"Unknown embed_style: {self.embed_style!r}")

    node_embedding = add_lap_out_of_place(node_embedding, inputs.get("lap_eigenvectors"))
    node_embedding = apply_global_conditioning(node_embedding, inputs)
    inputs["node_embedding"] = node_embedding

    if self.config.encoder_use_D_in_attn:
        # Off in their published conformer configs: this would feed the encoder
        # distances from the ground-truth conformer.
        conformer = inputs.get("conformer")
        distance = torch.cdist(conformer, conformer)
        mask = _gtmgc_modeling.make_cdist_mask(inputs.get("node_mask"))
        inputs["distance"] = _gtmgc_modeling.compute_distance_residual_bias(
            cdist=distance, cdist_mask=mask
        )

    attn_weight_dict: dict = {}
    for i, encoder_block in enumerate(self.encoder_blocks):
        block_out = encoder_block(**inputs)
        node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
        inputs["node_embedding"] = node_embedding
        attn_weight_dict[f"encoder_block_{i}"] = attn_weight
    return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}


def _patched_decoder_forward(self, **inputs):
    """Out-of-place equivalent of ``GTMGCDecoder.forward``."""
    node_embedding = add_lap_out_of_place(
        inputs.get("node_embedding"), inputs.get("lap_eigenvectors")
    )
    inputs["node_embedding"] = node_embedding

    attn_weight_dict: dict = {}
    for i, decoder_block in enumerate(self.decoder_blocks):
        block_out = decoder_block(**inputs)
        node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
        inputs["node_embedding"] = node_embedding
        attn_weight_dict[f"decoder_block_{i}"] = attn_weight
    return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}


def patch_inplace_lap_addition() -> None:
    """Replace the encoder/decoder forwards with autograd-safe versions."""
    if getattr(_gtmgc_modeling, "_step_up_inplace_patched", False):
        return
    _UPSTREAM_FORWARDS[_gtmgc_modeling.GTMGCEncoder] = _gtmgc_modeling.GTMGCEncoder.forward
    _UPSTREAM_FORWARDS[_gtmgc_modeling.GTMGCDecoder] = _gtmgc_modeling.GTMGCDecoder.forward
    _gtmgc_modeling.GTMGCEncoder.forward = _patched_encoder_forward
    _gtmgc_modeling.GTMGCDecoder.forward = _patched_decoder_forward
    _gtmgc_modeling._step_up_inplace_patched = True


def _conditioned_gtmgc_class() -> Any:
    """Define (once) the GTMGC subclass that adds charge / spin conditioning."""
    global _CONDITIONED_GTMGC
    if _CONDITIONED_GTMGC is not None:
        return _CONDITIONED_GTMGC

    class ConditionedGTMGC(_gtmgc.GTMGCForConformerPrediction):  # type: ignore[misc, valid-type]
        """GTMGC conditioned on molecule-level charge and spin."""

        def __init__(self, config: Any, n_global_features: int = N_GLOBAL_FEATURES) -> None:
            super().__init__(config)
            self.global_cond = global_condition_mlp(n_global_features, config.d_model)

        def forward(self, **inputs):
            global_features = inputs.get("global_features")
            if global_features is not None:
                inputs["global_embedding"] = self.global_cond(global_features.to(torch.float32))
            return super().forward(**inputs)

    _CONDITIONED_GTMGC = ConditionedGTMGC
    return ConditionedGTMGC


def get_gtmgc_collator():
    """Return the (lazily loaded) ``GTMGCCollator`` class."""
    _load_gtmgc()
    return _gtmgc.GTMGCCollator


def build_gtmgc_collator(conditioned: bool = False):
    """Collator instance for the DataLoader, optionally with charge/spin features."""
    base = get_gtmgc_collator()()
    return GlobalConditionCollator(base) if conditioned else base


def build_gtmgc(
    n_layers: int = 6,
    d_model: int = 256,
    d_ffn: int = 1024,
    n_head: int = 8,
    atom_vocab_size: int = 513,
    dropout: float = 0.0,
    embed_style: str = "atom_tokenized_ids",
    n_global_features: int = 0,
):
    """Instantiate GTMGC for conformer prediction.

    Defaults and attention flags follow their released QM9 checkpoint
    (``RichXuOvO/GTMGC-Qm9``): 6 + 6 layers, d_model 256, d_ffn 1024, 8 heads,
    Mole-BERT token embeddings. The encoder attends over the adjacency only, the
    decoder also over the distance matrix predicted by the first conformer head,
    so no ground-truth geometry reaches the encoder.
    """
    _load_gtmgc()
    config = _gtmgc.GTMGCConfig(
        n_encode_layers=n_layers,
        n_decode_layers=n_layers,
        encoder_use_A_in_attn=True,
        encoder_use_D_in_attn=False,
        decoder_use_A_in_attn=True,
        decoder_use_D_in_attn=True,
        embed_style=embed_style,
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
        d_ffn=d_ffn,
    )
    if n_global_features:
        return _conditioned_gtmgc_class()(config, n_global_features=n_global_features)
    return _gtmgc.GTMGCForConformerPrediction(config)


def _load_vendor_weights(model_cls: Any, config_cls: Any, checkpoint: str) -> Any:
    """Build a vendored model and load HuggingFace-hosted weights into it.

    ``PreTrainedModel.from_pretrained`` is not usable here: the vendored classes
    target transformers 4.32 and current transformers expects model APIs they
    don't implement. The two steps it would do for these plain checkpoints —
    build the config, load the state dict — are done directly instead.

    ``checkpoint`` is a Hub repo id or a local directory holding ``config.json``
    and ``pytorch_model.bin``.
    """
    local = Path(checkpoint)
    if local.is_dir():
        config_path, weights_path = local / "config.json", local / "pytorch_model.bin"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(checkpoint, "config.json"))
        weights_path = Path(hf_hub_download(checkpoint, "pytorch_model.bin"))

    with open(config_path) as handle:
        config = config_cls(**json.load(handle))
    model = model_cls(config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{checkpoint} is missing weights for: {sorted(missing)[:8]}")
    if unexpected:
        print(f"[gtmgc] ignoring {len(unexpected)} unexpected keys in {checkpoint}", flush=True)
    return model


def load_molebert_tokenizer(checkpoint: str = "RichXuOvO/MoleBERT-Tokenizer"):
    """Load the Mole-BERT atom tokenizer and its collator.

    GTMGC's conformer models embed the tokenizer's per-atom codebook indices.
    Returns ``(tokenizer, collator)``; the collator turns graph dicts into the
    PyG batch the tokenizer expects.
    """
    _load_gtmgc()
    tokenizer = _load_vendor_weights(
        _gtmgc.MoleBERTTokenizer, _gtmgc.MoleBERTTokenizerConfig, checkpoint
    )
    return tokenizer, _gtmgc.MoleBERTTokenizerCollator()


def load_pretrained_gtmgc(checkpoint: str):
    """Load one of their released checkpoints, e.g. ``RichXuOvO/GTMGC-Qm9``.

    Useful as a reference point: scoring their weights with our evaluation code
    should reproduce their published numbers.
    """
    _load_gtmgc()
    return _load_vendor_weights(_gtmgc.GTMGCForConformerPrediction, _gtmgc.GTMGCConfig, checkpoint)
