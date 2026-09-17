"""End-to-end checks of the training loop on the committed fixtures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from step_up import train as train_module
from step_up.train import TrainConfig, train


def _tiny_config(qm9_path: Path, tmp_path: Path, **overrides) -> TrainConfig:
    # (0.6, 0.2, 0.2) with split_seed 0 puts 11 / 4 / 5 of the 20 fixture rows in
    # train / val / test.
    fields = {
        "dataset_path": str(qm9_path),
        "dataset_source": "smiles",
        "split_ratios": (0.6, 0.2, 0.2),
        "n_layers": 1,
        "d_model": 32,
        "d_ffn": 64,
        "n_head": 4,
        "epochs": 1,
        "batch_size": 4,
        "eval_batch_size": 4,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
    }
    return TrainConfig(**{**fields, **overrides})


def test_train_writes_checkpoint_and_test_metrics(qm9_path, tmp_path) -> None:
    result = train(_tiny_config(qm9_path, tmp_path))
    assert result["best_epoch"] == 1
    assert math.isfinite(result["best_val_dmae"])
    assert (tmp_path / "run" / "best.pt").exists()
    metrics = json.loads((tmp_path / "run" / "test_metrics.json").read_text())
    assert math.isfinite(metrics["test_dmae"])
    assert metrics["best_epoch"] == 1


def test_train_rejects_empty_val_split(qm9_path, tmp_path) -> None:
    with pytest.raises(ValueError, match="Empty train or val split"):
        train(_tiny_config(qm9_path, tmp_path, split_ratios=(1.0, 0.0, 0.0)))


def test_epoch_with_every_batch_skipped_reports_nan() -> None:
    class NaNModel(torch.nn.Module):
        def forward(self, **batch):
            nan = torch.tensor(math.nan)
            return SimpleNamespace(loss=nan, cdist_mae=nan)

    loader = [{"node_mask": torch.ones(1, 3)}] * 2
    loss, dmae = train_module._run_epoch(NaNModel(), loader, None, "cpu", {"step": 0}, 1e-3, 1, 0)
    assert math.isnan(loss)
    assert math.isnan(dmae)


def test_train_fails_without_a_finite_validation_epoch(qm9_path, tmp_path, monkeypatch) -> None:
    real_run_epoch = train_module._run_epoch

    def nan_validation(model, loader, optimizer, *args, **kwargs):
        if optimizer is None:
            return math.nan, math.nan
        return real_run_epoch(model, loader, optimizer, *args, **kwargs)

    monkeypatch.setattr(train_module, "_run_epoch", nan_validation)
    with pytest.raises(RuntimeError, match="No epoch produced a finite validation D-MAE"):
        train(_tiny_config(qm9_path, tmp_path))
    assert not (tmp_path / "run" / "best.pt").exists()
    assert not (tmp_path / "run" / "test_metrics.json").exists()
