"""Confidence-weighted cell erasure with an adaptive per-step risk budget."""

import math

import torch
from torch.nn import functional as F


def sample_adaptive_dfm(model, coloring, *, risk_budget=0.05, tau=1.0,
                        seed=0, record_trajectory=False, progress=None):
    """Erase cells with sum(1-p_empty) <= risk_budget, using p_empty**(1/tau) order.

    Return CPU tokens, model-call counts, forced-step counts and optional history.
    The progress callback receives (deleted cells, total required deletions).
    """
    return _sample_adaptive_dfm(
        model, coloring, risk_budget=risk_budget, tau=tau, product_risk=False,
        seed=seed, record_trajectory=record_trajectory, progress=progress,
    )


def sample_product_risk_dfm(model, coloring, *, risk_budget=0.05, seed=0,
                           record_trajectory=False, progress=None):
    """Erase cells with 1-prod(p_empty) <= risk_budget, ordered by p_empty weights.

    Product risk uses a conditional-independence approximation. Stop with N
    colored cells. A zero-length prefix forces its first cell to be erased.
    Return CPU tokens, model-call counts, forced-step counts and optional history.
    The progress callback receives (deleted cells, total required deletions).
    """
    return _sample_adaptive_dfm(
        model, coloring, risk_budget=risk_budget, tau=1.0, product_risk=True,
        seed=seed, record_trajectory=record_trajectory, progress=progress,
    )


@torch.no_grad()
def _sample_adaptive_dfm(model, coloring, *, risk_budget, tau, product_risk,
                         seed, record_trajectory, progress):
    """Apply per-sample risk prefixes at time (N*N-M)/(N*N-N)."""
    tau, risk_budget = float(tau), float(risk_budget)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive.")
    if not math.isfinite(risk_budget) or risk_budget < 0:
        raise ValueError("risk_budget must be finite and nonnegative.")
    if product_risk and risk_budget > 1:
        raise ValueError("Product risk_budget must be between zero and one.")
    if product_risk:
        risk_limit = -math.log1p(-risk_budget) if risk_budget < 1 else math.inf
    else:
        risk_limit = risk_budget
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
            if product_risk:
                cumulative_score = (-log_p_empty).gather(1, order).cumsum(dim=1)
            else:
                cumulative_score = torch.sigmoid(-logits).gather(1, order).cumsum(dim=1)
            capacity = remaining[active] - size
            eligible = (cumulative_score <= risk_limit) & (ranks < capacity[:, None])
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
                selected_score = cumulative_score[0, deletions[0] - 1]
                estimated_risk = -torch.expm1(-selected_score) if product_risk else selected_score
                trajectory.append(dict(
                    state=state[0].cpu().clone(), iteration=int(iterations[0].item()),
                    time=(cells - count) / total_erasures, remaining=count,
                    deleted=int(deletions[0].item()),
                    estimated_risk=float(estimated_risk.item()),
                    forced=bool(forced[0].item()),
                ))
            if progress:
                progress(int((cells - remaining).sum().item()), len(state) * total_erasures)
    finally:
        model.train(was_training)
    return dict(tokens=state.cpu(), iterations=iterations.cpu(),
                forced_steps=forced_steps.cpu(), trajectory=trajectory)
