"""Inference-only confidence-weighted erasure with a fixed endpoint cell count."""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .queens_data import fixed_evaluation_splits
from .queens_dfm_inference import CHECKPOINT_FORMAT, QueensDFM


def load_trained_dfm(path, puzzles, *, device="cpu"):
    """Load trained DFM weights, dataset provenance and held-out indices."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Train and save the model in notebook 01 first: {path}")
    with path.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("format_version") != CHECKPOINT_FORMAT
            or saved.get("conditioning_mode") != "initial_state_only"):
        raise ValueError("Expected a Queens 8x8 DFM checkpoint with one erasure logit per cell.")
    if saved.get("dataset_sha256") != puzzles["metadata"]["data_sha256"]:
        raise ValueError("Checkpoint and dataset differ. Use the dataset from training.")
    if int(saved.get("global_step", 0)) < 1:
        raise ValueError("The checkpoint has not been trained.")
    splits = {name: values.cpu().numpy().astype(np.int64, copy=True)
              for name, values in saved["splits"].items()}
    all_indices = np.concatenate([splits[name] for name in ("train", "validation", "test")])
    if (len(all_indices) != len(puzzles["regions"])
            or not np.array_equal(np.sort(all_indices), np.arange(len(all_indices)))):
        raise ValueError("The checkpoint split does not partition the current dataset.")
    splits = fixed_evaluation_splits(splits, **saved.get("evaluation_config", {}))
    with torch.random.fork_rng(devices=[]):
        model = QueensDFM(**saved["model_config"])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    info = dict(
        checkpoint_sha256=checkpoint_sha256, checkpoint_path=str(path),
        dataset_sha256=saved["dataset_sha256"], seed=int(saved["seed"]),
        optimizer_updates=int(saved["global_step"]),
        epochs_completed=int(saved["history"]["epochs_completed"]),
        use_time_conditioning=model.use_time_conditioning,
        validation_puzzles=len(splits["validation"]), test_puzzles=len(splits["test"]),
    )
    return model, splits, info


@torch.no_grad()
def sample_confidence_dfm(model, coloring, *, steps=16, tau=1.0, seed=0,
                          capture_steps=(), progress=None):
    """Erase exactly N*N-N cells using p(EMPTY)**(1/tau) weighted sampling.

    Gumbel top-k samples without replacement. Cumulative quotas round
    (N*N-N)*k/steps to the nearest integer, with half-integers rounded up.
    All `steps` forward calls are made, including zero-deletion rounds when
    steps > N*N-N. The model receives only (current state, time).
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive.")
    device = next(model.parameters()).device
    state = torch.as_tensor(coloring, dtype=torch.long, device=device)
    if state.ndim == 2:
        state = state[None]
    if (state.ndim != 3 or len(state) == 0
            or tuple(state.shape[-2:]) != (model.board_size,) * 2):
        raise ValueError("Expected nonempty coloring with shape (B,N,N) or (N,N).")
    if bool(((state < 0) | (state >= model.board_size)).any()):
        raise ValueError("Start from a full coloring with no EMPTY cells.")
    state = state.clone().contiguous()
    capture_steps = set(capture_steps)
    if any(not isinstance(k, (int, np.integer)) or k < 0 or k > steps for k in capture_steps):
        raise ValueError("Capture steps must be integers between zero and steps.")
    snapshots = {0: state.cpu().clone()} if 0 in capture_steps else {}
    total_erasures = model.board_size**2 - model.board_size
    generator = torch.Generator(device=device).manual_seed(int(seed))
    was_training = model.training
    model.eval()
    erased = 0
    try:
        for step in range(steps):
            if model.use_time_conditioning:
                times = torch.full((len(state),), step / steps, device=device)
                logits = model(state, times)
            else:
                logits = model(state)
            log_p_empty = F.logsigmoid(logits[:, 0])
            cumulative = (2 * total_erasures * (step + 1) + steps) // (2 * steps)
            quota = cumulative - erased
            if quota:
                # Log probabilities avoid underflow when tau is small.
                uniforms = torch.rand(state.shape, device=device, generator=generator)
                limits = torch.finfo(uniforms.dtype)
                uniforms.clamp_(min=limits.tiny, max=1 - limits.eps)
                gumbels = -torch.log(-torch.log(uniforms))
                scores = (log_p_empty / tau + gumbels).flatten(1)
                scores.masked_fill_(state.flatten(1) == model.empty_token, -torch.inf)
                selected = scores.topk(quota, dim=1).indices
                state.view(len(state), -1).scatter_(1, selected, model.empty_token)
            erased = cumulative
            if step + 1 in capture_steps:
                snapshots[step + 1] = state.cpu().clone()
            if progress:
                progress(step + 1, steps)
    finally:
        model.train(was_training)
    return (state.cpu(), snapshots) if capture_steps else state.cpu()


def retry_statistics(successes, *, bootstrap_samples=3000, seed=0):
    """Whole-puzzle-bootstrap pass@k for (..., puzzle, attempt) outcomes."""
    successes = np.asarray(successes, dtype=bool)
    if successes.ndim < 2 or min(successes.shape) < 1 or bootstrap_samples < 1:
        raise ValueError("Expected nonempty (..., puzzle, attempt) outcomes.")
    n_puzzles, attempts = successes.shape[-2:]
    per_puzzle = np.zeros((*successes.shape[:-1], attempts + 1), dtype=float)
    failed = attempts - successes.sum(axis=-1)
    survival = np.ones_like(failed, dtype=float)
    for k in range(1, attempts + 1):
        survival *= np.maximum(failed - k + 1, 0) / (attempts - k + 1)
        per_puzzle[..., k] = 1 - survival
    mean = per_puzzle.mean(axis=-2)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_puzzles, np.full(n_puzzles, 1 / n_puzzles), size=bootstrap_samples)
    lower, upper = np.empty_like(mean), np.empty_like(mean)
    for configuration in np.ndindex(successes.shape[:-2]):
        draws = weights @ per_puzzle[configuration] / n_puzzles
        lower[configuration], upper[configuration] = np.quantile(draws, [0.025, 0.975], axis=0)
    return dict(attempts=np.arange(attempts + 1), mean=mean, lower=lower, upper=upper)
