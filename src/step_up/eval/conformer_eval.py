"""Conformer metrics in ReBind's published protocol.

``external/ReBIND/evaluate.py`` reports three numbers, and the details matter if
you want to compare against the paper:

- **D-MAE / D-RMSE** pool absolute / squared errors over *every* pairwise
  distance in the split and divide by the total count of distances, including
  the zero diagonal. This is a distance-weighted average, so larger molecules
  count more — unlike the per-batch mean the training loop prints.
- **C-RMSD** is RDKit's ``GetBestRMS`` (optimal alignment, symmetry aware) on
  the molecule with hydrogens removed, averaged per molecule.

For reference, ReBind's paper reports test D-MAE 0.254, D-RMSE 0.446 and
C-RMSD 0.321 on QM9.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem, rdMolAlign
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def kabsch_rmsd(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMSD after optimally aligning ``pred`` onto ``target`` (Kabsch, no symmetry matching).

    Both are ``(n, 3)`` over an individual molecule's real atoms. Doing the
    alignment here rather than trusting the model's own matters: ReBind and GTMGC
    align inside the prediction head over the *padded* batch tensor, so padding
    zeros pull the centroid and rotation off and inflate the RMSD — badly for a
    small molecule batched with a large one.
    """
    p = pred - pred.mean(dim=0, keepdim=True)
    q = target - target.mean(dim=0, keepdim=True)
    u, _, vt = torch.linalg.svd(p.T @ q)
    # Flip the last axis if the optimal rotation came out as a reflection.
    sign = torch.sign(torch.det(vt.T @ u.T))
    correction = torch.diag(torch.tensor([1.0, 1.0, float(sign)], dtype=p.dtype))
    rotation = vt.T @ correction @ u.T
    aligned = (rotation @ p.T).T
    return float(((aligned - q) ** 2).sum(dim=-1).mean().sqrt())


def _predicted_mol(mol: Chem.Mol, coords: Sequence[Sequence[float]]) -> Chem.Mol:
    """Copy ``mol`` and replace its conformer with ``coords``."""
    mol_hat = Chem.Mol(mol)
    conformer = rdchem.Conformer(mol.GetNumAtoms())
    for i, position in enumerate(coords):
        conformer.SetAtomPosition(i, [float(v) for v in position])
    mol_hat.RemoveAllConformers()
    mol_hat.AddConformer(conformer)
    return mol_hat


def evaluate_split(
    model: torch.nn.Module,
    dataset: Any,
    subset: Subset,
    collator: Any,
    device: str = "cpu",
    batch_size: int = 100,
    remove_hs: bool = True,
    rmsd_method: str = "auto",
) -> dict[str, float | str]:
    """Score ``subset`` with ReBind's metric definitions.

    ``dataset`` must expose ``rdkit_mol(idx)`` for the C-RMSD part; the indices
    in ``subset`` index into it.

    ``rmsd_method`` is ``"auto"`` (RDKit ``GetBestRMS`` where a molecule can be
    built, else aligned coordinates) or ``"aligned_coords"`` to force the plain
    metric everywhere, which is what makes RMSD comparable between the organic
    and organometallic sets.
    """
    if rmsd_method not in ("auto", "aligned_coords"):
        raise ValueError(f"Unknown rmsd_method: {rmsd_method!r}")
    model.eval()
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    total_abs, total_sq, total_dist = 0.0, 0.0, 0
    total_rmsd, n_rmsd, rmsd_failures, n_mol = 0.0, 0, 0, 0
    # What we ask for vs what we end up reporting: "auto" degrades to aligned
    # coordinates for rows that have no RDKit molecule (the MOL2 path).
    use_rdkit = rmsd_method == "auto"
    reported_method = "rdkit_bestrms" if use_rdkit else "aligned_coords"

    position = 0
    for batch in tqdm(loader, desc="evaluating", leave=False, dynamic_ncols=True):
        device_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**device_batch)
        node_mask = device_batch["node_mask"].bool()
        for row in range(node_mask.shape[0]):
            keep = node_mask[row]
            pred = out.conformer_hat[row][keep].double().cpu()
            target = out.conformer[row][keep].double().cpu()
            d_pred, d_true = torch.cdist(pred, pred), torch.cdist(target, target)
            delta = (d_pred - d_true).abs()
            total_abs += float(delta.sum())
            total_sq += float((delta**2).sum())
            total_dist += delta.numel()  # n*n, diagonal included, as in ReBind
            n_mol += 1

            mol = None
            if use_rdkit:
                try:
                    mol = dataset.rdkit_mol(subset.indices[position + row])
                except NotImplementedError:
                    # Organometallic (MOL2) rows never build an RDKit molecule.
                    reported_method = "aligned_coords"
            if mol is None:
                # Optimal rigid alignment on this molecule's real atoms. Not
                # symmetry matched, so only compare to other aligned_coords numbers.
                total_rmsd += kabsch_rmsd(pred, target)
                n_rmsd += 1
            else:
                ref = _predicted_mol(mol, target.tolist())
                probe = _predicted_mol(mol, pred.tolist())
                try:
                    if remove_hs:
                        ref, probe = Chem.RemoveHs(ref), Chem.RemoveHs(probe)
                    total_rmsd += rdMolAlign.GetBestRMS(probe, ref)
                    n_rmsd += 1
                except Exception:
                    # GetBestRMS can fail on molecules RDKit won't match up.
                    rmsd_failures += 1
        position += node_mask.shape[0]

    metrics: dict[str, float | str] = {
        "d_mae": total_abs / max(total_dist, 1),
        "d_rmse": (total_sq / max(total_dist, 1)) ** 0.5,
        "c_rmsd": total_rmsd / max(n_rmsd, 1),
        "c_rmsd_method": reported_method,
        "n_molecules": float(n_mol),
        "n_rmsd_molecules": float(n_rmsd),
        "n_rmsd_failures": float(rmsd_failures),
        "remove_hs": float(remove_hs),
    }
    return metrics


def write_metrics(metrics: dict[str, float | str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
