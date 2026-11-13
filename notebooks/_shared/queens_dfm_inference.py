"""Queens DFM with the coloring as its initial state only."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import Denoiser
from .queens_data import BOARD_SIZE


class QueensDFM(nn.Module):
    """DFM conditioned through its initial state only: forward(state, times).

    Input: nine current-state channels + two coordinates. Both the network
    features and the absorbing-path support mask depend only on the current
    state, never on an archived original coloring.
    """

    conditioning_mode = "initial_state_only"

    def __init__(self, board_size=BOARD_SIZE, base_channels=64, time_dim=128,
                 time_embedding_scale=100.0):
        super().__init__()
        if board_size != BOARD_SIZE:
            raise ValueError("This checkpoint family uses 8x8 Queens boards.")
        self.board_size = int(board_size)
        self.num_classes = self.board_size + 1
        self.empty_token = self.board_size
        self.time_embedding_scale = float(time_embedding_scale)
        self.network = Denoiser(
            embedding_dim=self.num_classes + 2, output_dim=self.num_classes,
            base_channels=base_channels, time_dim=time_dim,
        )
        axis = torch.linspace(-1, 1, self.board_size)
        rr, cc = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack((rr, cc))[None])

    def forward(self, state, times):
        channels = F.one_hot(state.long(), self.num_classes).permute(0, 3, 1, 2).float()
        inputs = torch.cat((channels, self.coordinates.expand(len(state), -1, -1, -1)), 1)
        logits = self.network(inputs, times * self.time_embedding_scale)
        # A colored cell can retain its CURRENT color or become EMPTY. An empty
        # cell remains empty. No original-color lookup is used for erased cells.
        allowed = F.one_hot(state.long(), self.num_classes).permute(0, 3, 1, 2).bool()
        allowed[:, self.empty_token] = True
        return logits.masked_fill(~allowed, -torch.inf)


@torch.no_grad()
def sample_queens_dfm(candidate, coloring, *, steps=100, seed=0, progress=None,
                      capture_steps=()):
    """Initialize from coloring, then follow the model's declared conditioning.

    Return tokens, or (tokens, snapshots) when capture_steps is nonempty.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    device = next(candidate.parameters()).device
    coloring = torch.as_tensor(coloring, dtype=torch.long, device=device)
    if coloring.ndim == 2:
        coloring = coloring[None]
    if coloring.ndim != 3 or tuple(coloring.shape[-2:]) != (candidate.board_size,) * 2:
        raise ValueError("Expected coloring with shape (B,8,8) or (8,8).")
    if bool(((coloring < 0) | (coloring >= candidate.board_size)).any()):
        raise ValueError("Unexpected region IDs.")
    state = coloring.clone()
    capture_steps = set(map(int, capture_steps))
    if any(step < 0 or step > steps for step in capture_steps):
        raise ValueError("Capture steps must be between zero and steps.")
    snapshots = {0: state.cpu().clone()} if 0 in capture_steps else {}
    generator = torch.Generator(device=device).manual_seed(int(seed))
    was_training = candidate.training
    candidate.eval()
    try:
        for step in range(steps):
            times = torch.full((len(state),), step / steps, device=device)
            logits = candidate(state, times)
            p_empty = logits.softmax(dim=1)[:, candidate.empty_token]
            jump = torch.rand(state.shape, device=device, generator=generator) < p_empty / (steps - step)
            state = torch.where(jump & (state != candidate.empty_token), candidate.empty_token, state)
            if step + 1 in capture_steps:
                snapshots[step + 1] = state.cpu().clone()
            if progress:
                progress(step + 1, steps)
    finally:
        candidate.train(was_training)
    return (state.cpu(), snapshots) if capture_steps else state.cpu()
