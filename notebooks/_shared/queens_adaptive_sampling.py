"""Confidence-weighted cell erasure with an adaptive per-step risk budget."""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def sample_adaptive_dfm(model, coloring, *, risk_budget=0.05, tau=1.0,
                        seed=0, record_trajectory=False, progress=None):
    """Sample until N cells remain; each trajectory uses at most N*N-N calls.

    Rank colored cells by Gumbel scores log(p_empty)/tau + Gumbel(0,1).
    Erase the longest prefix with sum(1-p_empty) <= risk_budget, capped at
    M-N cells. A zero-length prefix forces one deletion to ensure progress.
    Each sample uses its own time t=(N*N-M)/(N*N-N).

    Return CPU tensors `tokens`, `iterations` and `forced_steps`. With
    record_trajectory=True, also record each state of a single-sample run.
    The progress callback receives (deleted cells, total deletions).
    """
    tau, risk_budget = float(tau), float(risk_budget)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive.")
    if not math.isfinite(risk_budget) or risk_budget < 0:
        raise ValueError("risk_budget must be finite and nonnegative.")
    device = next(model.parameters()).device
    state = torch.as_tensor(coloring, dtype=torch.long, device=device)
    if state.ndim == 2:
        state = state[None]
    size = model.board_size
    if state.ndim != 3 or len(state) == 0 or tuple(state.shape[-2:]) != (size, size):
        raise ValueError("Expected a nonempty (B,N,N) or (N,N) coloring.")
    if bool(((state < 0) | (state >= size)).any()):
        raise ValueError("Start from a complete region coloring.")
    if record_trajectory and len(state) != 1:
        raise ValueError("Record one trajectory at a time.")
    state = state.clone().contiguous()
    cells, total_erasures = size**2, size**2 - size
    if total_erasures < 1:
        raise ValueError("The board must have at least two rows.")
    remaining = torch.full((len(state),), cells, device=device, dtype=torch.long)
    iterations = torch.zeros_like(remaining)
    forced_steps = torch.zeros_like(remaining)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    ranks = torch.arange(cells, device=device)[None]
    trajectory = []
    if record_trajectory:
        trajectory.append(dict(state=state[0].cpu().clone(), iteration=0, time=0.0,
                               remaining=cells, deleted=0, estimated_risk=0.0, forced=False))
    was_training = model.training
    model.eval()
    try:
        for _ in range(total_erasures):
            active = torch.nonzero(remaining > size, as_tuple=False).flatten()
            if not len(active):
                break
            current = state[active]
            times = (cells - remaining[active]).float() / total_erasures
            logits = model(current, times)[:, 0].flatten(1)
            log_p_empty = F.logsigmoid(logits)
            uniforms = torch.rand(log_p_empty.shape, device=device, generator=generator)
            limits = torch.finfo(uniforms.dtype)
            uniforms.clamp_(min=limits.tiny, max=1 - limits.eps)
            scores = log_p_empty / tau - torch.log(-torch.log(uniforms))
            scores.masked_fill_(current.flatten(1) == model.empty_token, -torch.inf)
            order = scores.argsort(dim=1, descending=True)
            risks = torch.sigmoid(-logits).gather(1, order)
            cumulative_risk = risks.cumsum(dim=1)
            capacity = remaining[active] - size
            eligible = (cumulative_risk <= risk_budget) & (ranks < capacity[:, None])
            deletions = eligible.sum(dim=1)
            forced = deletions == 0
            deletions = deletions.clamp_min(1)
            remove = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                1, order, ranks < deletions[:, None],
            )
            current.flatten(1).masked_fill_(remove, model.empty_token)
            state[active] = current
            remaining[active] -= deletions
            iterations[active] += 1
            forced_steps[active] += forced.long()
            if record_trajectory:
                count = int(remaining[0].item())
                trajectory.append(dict(
                    state=state[0].cpu().clone(), iteration=int(iterations[0].item()),
                    time=(cells - count) / total_erasures, remaining=count,
                    deleted=int(deletions[0].item()),
                    estimated_risk=float(cumulative_risk[0, deletions[0] - 1].item()),
                    forced=bool(forced[0].item()),
                ))
            if progress:
                progress(int((cells - remaining).sum().item()), len(state) * total_erasures)
    finally:
        model.train(was_training)
    return dict(tokens=state.cpu(), iterations=iterations.cpu(),
                forced_steps=forced_steps.cpu(), trajectory=trajectory)
