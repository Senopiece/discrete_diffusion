"""Queens DFM endpoint erasure logits with optional time conditioning."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import Denoiser
from .queens_data import BOARD_SIZE


CHECKPOINT_FORMAT = "queens8_erasure_dfm_v1"


class QueensDFM(nn.Module):
    """Predict one endpoint EMPTY logit per cell, shaped (B,1,N,N).

    Inputs are nine one-hot state channels and two coordinate channels.
    Optional time conditioning adds a learned timestep branch.
    Sigmoid converts each output logit to an endpoint erasure probability.
    """

    conditioning_mode = "initial_state_only"

    def __init__(self, board_size=BOARD_SIZE, base_channels=64, time_dim=128,
                 time_embedding_scale=100.0, use_time_conditioning=True):
        super().__init__()
        if board_size != BOARD_SIZE:
            raise ValueError("This checkpoint family uses 8x8 Queens boards.")
        self.board_size = int(board_size)
        self.num_states = self.board_size + 1
        self.empty_token = self.board_size
        self.use_time_conditioning = bool(use_time_conditioning)
        if self.use_time_conditioning:
            self.time_embedding_scale = float(time_embedding_scale)
        self.network = Denoiser(
            embedding_dim=self.num_states + 2, output_dim=1,
            base_channels=base_channels, time_dim=time_dim,
            use_time_conditioning=self.use_time_conditioning,
        )
        axis = torch.linspace(-1, 1, self.board_size)
        rr, cc = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack((rr, cc))[None])

    def forward(self, state, times=None):
        channels = F.one_hot(state.long(), self.num_states).permute(0, 3, 1, 2).float()
        inputs = torch.cat((channels, self.coordinates.expand(len(state), -1, -1, -1)), 1)
        if self.use_time_conditioning:
            if times is None:
                raise ValueError("Times are required when time conditioning is enabled.")
            return self.network(inputs, times * self.time_embedding_scale)
        return self.network(inputs)


@torch.no_grad()
def sample_queens_dfm(candidate, coloring, *, steps=100, seed=0, progress=None,
                      capture_steps=()):
    """Sample the categorical flow from a complete region coloring.

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
            if candidate.use_time_conditioning:
                times = torch.full((len(state),), step / steps, device=device)
                logits = candidate(state, times)
            else:
                logits = candidate(state)
            p_empty = logits[:, 0].sigmoid()
            jump = torch.rand(state.shape, device=device, generator=generator) < p_empty / (steps - step)
            state = torch.where(jump & (state != candidate.empty_token), candidate.empty_token, state)
            if step + 1 in capture_steps:
                snapshots[step + 1] = state.cpu().clone()
            if progress:
                progress(step + 1, steps)
    finally:
        candidate.train(was_training)
    return (state.cpu(), snapshots) if capture_steps else state.cpu()
