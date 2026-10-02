"""The vendored GTMGC wrapper."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from step_up.data.csv_dataset import CSVMoleculeDataset
from step_up.models import build_model, build_model_collator
from step_up.models import gtmgc as gtmgc_module


def _tokenized_batch(qm9_sdf_path, tokens_path, n: int = 4) -> dict:
    ds = CSVMoleculeDataset(qm9_sdf_path, "sdf", id_column="mol_id", token_file=tokens_path)
    loader = DataLoader(
        Subset(ds, list(range(n))), batch_size=n, collate_fn=build_model_collator("gtmgc")
    )
    return next(iter(loader))


def test_tokens_reach_the_batch_as_node_input_ids(qm9_sdf_path, qm9_sdf_tokens_path) -> None:
    """GTMGC embeds Mole-BERT ids, so the collator must receive them from the dataset."""
    ds = CSVMoleculeDataset(qm9_sdf_path, "sdf", id_column="mol_id", token_file=qm9_sdf_tokens_path)
    methane = ds[0]
    assert len(methane["input_ids"]) == methane["num_nodes"]

    batch = _tokenized_batch(qm9_sdf_path, qm9_sdf_tokens_path)
    assert "node_input_ids" in batch
    # The collator shifts ids by 1 so 0 can mean padding.
    assert batch["node_input_ids"][0][0].item() == methane["input_ids"][0] + 1
    assert batch["node_input_ids"].max().item() <= 512


def test_forward_and_backward_in_fp32(qm9_sdf_path, qm9_sdf_tokens_path) -> None:
    batch = _tokenized_batch(qm9_sdf_path, qm9_sdf_tokens_path)
    torch.manual_seed(0)
    model = build_model("gtmgc", n_layers=2, d_model=64, d_ffn=128, n_head=4)
    model.train()
    out = model(**batch)
    assert torch.isfinite(out.loss)
    assert out.conformer_hat.shape == out.conformer.shape
    # Upstream's in-place Laplacian write breaks this; our patch is what makes fp32 work.
    out.loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and (p.grad != 0).any()
        for p in model.parameters()
    )


def test_patched_forward_matches_upstream(qm9_sdf_path, qm9_sdf_tokens_path, monkeypatch) -> None:
    """The out-of-place patch must not change what the model computes."""
    batch = _tokenized_batch(qm9_sdf_path, qm9_sdf_tokens_path)
    torch.manual_seed(0)
    model = build_model("gtmgc", n_layers=2, d_model=64, d_ffn=128, n_head=4)
    model.eval()
    with torch.no_grad():
        patched = model(**batch)
        for cls, upstream_forward in gtmgc_module._UPSTREAM_FORWARDS.items():
            monkeypatch.setattr(cls, "forward", upstream_forward)
        upstream = model(**batch)
    torch.testing.assert_close(patched.loss, upstream.loss)
    torch.testing.assert_close(patched.conformer_hat, upstream.conformer_hat)


def test_atom_type_embedding_needs_no_tokenizer(qm9_sdf_path) -> None:
    """Their ablation embed_style runs off atom types, which we always have."""
    ds = CSVMoleculeDataset(qm9_sdf_path, "sdf")
    loader = DataLoader(Subset(ds, [0, 1]), batch_size=2, collate_fn=build_model_collator("gtmgc"))
    torch.manual_seed(0)
    model = gtmgc_module.build_gtmgc(
        n_layers=1, d_model=32, d_ffn=64, n_head=4, embed_style="atom_type_ids", atom_vocab_size=119
    )
    model.eval()
    with torch.no_grad():
        out = model(**next(iter(loader)))
    assert torch.isfinite(out.loss)
