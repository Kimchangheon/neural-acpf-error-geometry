"""Fine-tuning loop helper for GridSFM. OPFData format only.

`finetune_opfdata(model, train_loader, val_loader, epochs, ...)` runs the
standard FT recipe: AdamW on the full model + `compute_loss` with the
default lambda config + grad clip 5.0 + per-epoch val pass.

`train_loader` and `val_loader` must yield batches in OPFData's HeteroData
schema (see `OPFDataAdapterDataset` and `SyntheticMixedDataset`). Custom
data formats are not supported.

Per-epoch returns a list of `{epoch, train_loss, val_loss, val_cost_mape,
elapsed_s, ...}` dicts so callers can plot loss curves.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .eval import eval_pass
from .loss import compute_loss

_logger = logging.getLogger(__name__)


def finetune_opfdata(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    loss_kwargs: Optional[Dict[str, float]] = None,
    on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Fine-tune `model` on `train_loader` (OPFData format) for `epochs` epochs.

    Args:
      model: a `GridTransformerBackbone` (or any module with the same
        forward + prediction-attach contract).
      train_loader: yields HeteroData batches in the OPFData schema with
        `feasible`, `feas_logit` targets and the per-node-type `.y` labels.
      val_loader: optional held-out loader for per-epoch eval. If `None`,
        the returned log carries only `train_loss`.
      epochs, lr, weight_decay, grad_clip: standard knobs.
      loss_kwargs: passed through to `compute_loss` (lambda overrides).
        The same overrides flow into the per-epoch val pass so train and
        val losses remain directly comparable.
      on_epoch_end: optional callback invoked with the per-epoch log dict
        after each epoch (e.g. for live plotting).

    Returns a list of per-epoch dicts.

    Note: if `train_loader` wraps a `SyntheticMixedDataset` and uses a
    DataLoader with `persistent_workers=True`, the per-worker dataset
    copies retain the same `_epoch` across epochs and `set_epoch(ep)`
    will not reach them; the perturbation seeding will not roll between
    epochs (synth variance falls to 0). Use `persistent_workers=False`
    when you want fresh perturbations each epoch.
    """
    device = next(model.parameters()).device
    train_ds = train_loader.dataset
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_kwargs = loss_kwargs or {}

    log: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep in range(epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(ep)

        model.train()
        sum_loss, n_iters, n_skipped = 0.0, 0, 0
        ep_start = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            model(batch)
            loss, _ = compute_loss(batch, **loss_kwargs)
            if not torch.isfinite(loss):
                # Non-finite loss would corrupt every parameter on opt.step.
                _logger.warning("epoch %d iter %d: non-finite loss; "
                                "skipping backward+step",
                                ep, n_iters + n_skipped)
                n_skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            sum_loss += float(loss.item())
            n_iters += 1
        if n_skipped:
            _logger.warning("epoch %d: %d/%d batches skipped due to non-finite loss",
                            ep, n_skipped, n_iters + n_skipped)
        # NaN (not 0.0) when no batch contributed — 0.0 would falsely
        # signal a perfectly-converged epoch in the loss plot.
        if n_iters == 0:
            _logger.warning("epoch %d: 0 batches contributed; "
                            "train_loss reported as NaN", ep)
            train_loss_avg = float("nan")
        else:
            train_loss_avg = sum_loss / n_iters

        entry: Dict[str, Any] = {
            "epoch": ep,
            "train_loss": train_loss_avg,
            "n_train_iters": n_iters,
            "n_train_skipped": n_skipped,
            "epoch_s": round(time.time() - ep_start, 1),
            "elapsed_s": round(time.time() - t0, 1),
        }
        if val_loader is not None:
            val = eval_pass(model, val_loader, device=device, loss_kwargs=loss_kwargs)
            entry.update({f"val_{k}": v for k, v in val.items()})
        log.append(entry)
        if on_epoch_end is not None:
            on_epoch_end(entry)

    model.eval()
    return log
