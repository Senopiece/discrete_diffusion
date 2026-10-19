"""Checkpoint-compatible Queens DFM inference, independent of notebook globals."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import Denoiser
from .queens_data import BOARD_SIZE


class QueensDFM(nn.Module):
    """The coloring-to-colored-queens model from the DFM notebook (v1)."""

    def __init__(self, board_size=7, base_channels=64, time_dim=128,
                 time_embedding_scale=100.0):
        super().__init__()
        if board_size != BOARD_SIZE:
            raise ValueError("This checkpoint family uses 7×7 Queens boards.")
        self.board_size = int(board_size)
        self.num_classes = self.board_size + 1
        self.empty_token = self.board_size
        self.time_embedding_scale = float(time_embedding_scale)
        self.network = Denoiser(
            embedding_dim=self.num_classes + self.board_size + 2,
            output_dim=self.num_classes, base_channels=base_channels, time_dim=time_dim,
        )
        positions = torch.linspace(-1, 1, self.board_size)
        rr, cc = torch.meshgrid(positions, positions, indexing="ij")
        self.register_buffer("coordinates", torch.stack((rr, cc))[None])

    def forward(self, state, coloring, times):
        state_channels = F.one_hot(state.long(), self.num_classes).permute(0, 3, 1, 2).float()
        color_channels = F.one_hot(coloring.long(), self.board_size).permute(0, 3, 1, 2).float()
        inputs = torch.cat((
            state_channels, color_channels,
            self.coordinates.expand(len(state), -1, -1, -1),
        ), dim=1)
        logits = self.network(inputs, times * self.time_embedding_scale)
        allowed = F.one_hot(coloring.long(), self.num_classes).permute(0, 3, 1, 2).bool()
        allowed = allowed & (state != self.empty_token)[:, None]
        allowed[:, self.empty_token] = True
        return logits.masked_fill(~allowed, -torch.inf)


def load_queens_dfm(path, *, dataset_sha256, device="cpu"):
    """Load only inference weights; return provenance and the checkpoint test split."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != "queens7_dfm_v1":
        raise ValueError("Expected a queens7_dfm_v1 checkpoint.")
    if checkpoint.get("dataset_sha256") != dataset_sha256:
        raise ValueError("DFM checkpoint and current dataset differ.")
    # Construct on CPU without changing the active D3PM training RNG state.
    with torch.random.fork_rng(devices=[]):
        candidate = QueensDFM(**checkpoint["model_config"])
    candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    candidate.to(device).eval()
    info = dict(
        checkpoint_path=str(path), checkpoint_sha256=digest.hexdigest(),
        dataset_sha256=dataset_sha256, flow_steps=int(checkpoint["flow_steps"]),
        epochs=int(checkpoint["history"]["epochs_completed"]),
        global_step=int(checkpoint["global_step"]), seed=int(checkpoint["seed"]),
        test_indices=np.asarray(checkpoint["splits"]["test"].cpu(), dtype=np.int64).copy(),
    )
    return candidate, info


@torch.no_grad()
def sample_queens_dfm(candidate, coloring, *, steps=100, seed=0, progress=None):
    """Same irreversible categorical sampler as the DFM notebook; return tokens."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    device = next(candidate.parameters()).device
    coloring = torch.as_tensor(coloring, dtype=torch.long, device=device)
    if coloring.ndim == 2:
        coloring = coloring[None]
    if coloring.ndim != 3 or tuple(coloring.shape[-2:]) != (candidate.board_size,) * 2:
        raise ValueError("Expected coloring with shape (B,7,7) or (7,7).")
    if bool(((coloring < 0) | (coloring >= candidate.board_size)).any()):
        raise ValueError("Unexpected region IDs.")
    state = coloring.clone()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    was_training = candidate.training
    candidate.eval()
    try:
        for step in range(steps):
            times = torch.full((len(state),), step / steps, device=device)
            p_empty = candidate(state, coloring, times).softmax(dim=1)[:, candidate.empty_token]
            jump = torch.rand(state.shape, device=device, generator=generator) < p_empty / (steps - step)
            state = torch.where(jump & (state != candidate.empty_token), candidate.empty_token, state)
            if progress:
                progress(step + 1, steps)
    finally:
        candidate.train(was_training)
    return state.cpu()
