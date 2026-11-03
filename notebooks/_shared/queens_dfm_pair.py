"""Compare DFM with repeated coloring conditioning vs initial-state-only DFM."""

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .queens_data import solution_checks
from .queens_dfm_inference import load_queens_dfm, sample_queens_dfm


def _evaluate(candidate, info, puzzles, indices, *, steps, attempts, seed, batch_size,
              cache_path, fallback_paths=(), refresh=False, progress=None, label=""):
    for path in [Path(cache_path), *map(Path, fallback_paths)]:
        if refresh or not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            required = {"dataset_indices", "successes", "checkpoint_sha256", "dataset_sha256",
                        "steps", "attempts", "seed", "batch_size", "sampler_version"}
            if required.issubset(saved.files) and (
                str(saved["checkpoint_sha256"]) == info["checkpoint_sha256"]
                and str(saved["dataset_sha256"]) == puzzles["metadata"]["data_sha256"]
                and np.array_equal(saved["dataset_indices"], indices)
                and saved["successes"].shape == (len(indices), attempts)
                and int(saved["steps"]) == steps and int(saved["attempts"]) == attempts
                and int(saved["seed"]) == seed and int(saved["batch_size"]) == batch_size
                and str(saved["sampler_version"]) == info["sampler_version"]
            ):
                if progress:
                    progress(label, len(indices) * attempts * steps, len(indices) * attempts * steps, "cached")
                return saved["successes"].astype(bool), str(path)
    flat = np.repeat(indices, attempts)
    outcomes = []
    for start in range(0, len(flat), batch_size):
        batch_indices = flat[start:start + batch_size]

        def update(done, total):
            if progress and (done % 5 == 0 or done == total):
                progress(label, start * steps + done * len(batch_indices), len(flat) * steps, "sampling")

        predicted = sample_queens_dfm(
            candidate, puzzles["regions"][batch_indices], steps=steps,
            seed=seed + start, progress=update,
        ).numpy()
        outcomes.append(solution_checks(puzzles["regions"][batch_indices], predicted)["solved"])
    successes = np.concatenate(outcomes).reshape(len(indices), attempts)
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, dataset_indices=indices, successes=successes,
        checkpoint_sha256=info["checkpoint_sha256"], dataset_sha256=info["dataset_sha256"],
        steps=steps, attempts=attempts, seed=seed, batch_size=batch_size,
        sampler_version=info["sampler_version"],
        training_epochs=info["epochs"], training_optimizer_steps=info["global_step"],
    )
    return successes, str(path)


def _pass_at_k(successes):
    return _pass_at_k_from_counts(successes.sum(axis=1), successes.shape[1])


def _pass_at_k_from_counts(counts, attempts):
    """Average success over every k-subset of the observed independent attempts."""
    counts = np.asarray(counts, dtype=int)
    if attempts < 1 or counts.ndim != 1 or np.any((counts < 0) | (counts > attempts)):
        raise ValueError("Expected success counts between zero and the attempt count.")
    values = np.zeros((len(counts), attempts + 1))
    failures = attempts - counts
    probability_none = np.ones(len(counts))
    for k in range(1, attempts + 1):
        probability_none *= np.maximum(failures - k + 1, 0) / (attempts - k + 1)
        values[:, k] = 1 - probability_none
    return values


def solvability_retry_scatter(comparison, puzzles):
    """Paired per-puzzle pass@k scatter with a native Plotly attempt slider.

    P(at least one success in k attempts) = 1-C(A-c,k)/C(A,k).
    Uses recorded counts only; moving the slider never evaluates a network.
    """
    import plotly.graph_objects as go

    if comparison["dataset_sha256"] != puzzles["metadata"]["data_sha256"]:
        raise ValueError("The comparison belongs to another puzzle dataset.")
    models = comparison["models"]
    if len(models) != 2:
        raise ValueError("Expected two compared models.")
    attempts = int(comparison["attempts"])
    indices = np.asarray(comparison["dataset_indices"], dtype=int)
    curves = [_pass_at_k_from_counts(model["counts"], attempts) for model in models]
    if any(len(curve) != len(indices) for curve in curves):
        raise ValueError("Puzzle IDs and model success counts do not match.")
    sources = puzzles["source_names"][indices]
    groups = [(source, np.flatnonzero(sources == source)) for source in sorted(set(sources.tolist()))]
    colors = {"Synthetic": "#5267DB", "Queens.game": "#D88437",
              "AMAZE": "#2F9D88", "Queendom": "#9B6BD3"}

    def axis_title(model, k):
        return f"{model['label']} · solved within {k} attempts (%)"

    def title(k):
        return f"DFM conditioning · at least one solution in {k} attempts"

    def source_trace(source, selected, k):
        return go.Scatter(
            x=100 * curves[0][selected, k], y=100 * curves[1][selected, k],
            ids=[str(int(indices[i])) for i in selected], mode="markers", name=source,
            marker=dict(size=9, opacity=.7, color=colors.get(source, "#64748B")),
            customdata=[
                [int(indices[i]), str(puzzles["source_ids"][indices[i]]),
                 int(models[0]["counts"][i]), int(models[1]["counts"][i]), k, attempts]
                for i in selected
            ],
            hovertemplate=(
                "Puzzle #%{customdata[0]} · %{customdata[1]}"
                "<br>At least once in %{customdata[4]} attempts"
                f"<br>{models[0]['label']}: %{{x:.2f}}%"
                f"<br>{models[1]['label']}: %{{y:.2f}}%"
                "<br>Recorded single-attempt successes: "
                "%{customdata[2]}/%{customdata[5]} vs %{customdata[3]}/%{customdata[5]}"
                "<extra>%{fullData.name}</extra>"
            ),
        )

    fig = go.Figure(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines", name="Equal solvability",
        line=dict(color="#A5ADBA", dash="dash", width=1.5), hoverinfo="skip"))
    for source, selected in groups:
        fig.add_trace(source_trace(source, selected, 1))
    fig.frames = [go.Frame(
        name=str(k), data=[source_trace(source, selected, k) for source, selected in groups],
        traces=list(range(1, len(groups) + 1)),
        layout=go.Layout(title=dict(text=title(k)),
                         xaxis=dict(title=axis_title(models[0], k)),
                         yaxis=dict(title=axis_title(models[1], k))))
        for k in range(1, attempts + 1)]
    fig.update_layout(
        title=dict(text=title(1), x=0), template="plotly_white", autosize=True, height=700,
        xaxis=dict(title=axis_title(models[0], 1), range=[-2, 102], constrain="domain"),
        yaxis=dict(title=axis_title(models[1], 1), range=[-2, 102],
                   scaleanchor="x", scaleratio=1, constrain="domain"),
        sliders=[dict(
            active=0, x=.025, len=.95, y=-.17, yanchor="top",
            currentvalue=dict(prefix="Attempts: ", suffix=f" / {attempts}", font=dict(size=16)),
            pad=dict(t=25, b=15), ticklen=5, font=dict(size=11),
            steps=[dict(label=str(k), method="animate", args=[
                [str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                               transition=dict(duration=0))]) for k in range(1, attempts + 1)]
        )],
        legend=dict(orientation="h", x=0, y=-.40),
        margin=dict(l=85, r=25, t=80, b=185), font=dict(family="Arial, sans-serif", size=15),
    )
    return fig


def compare_dfm_conditioning(puzzles, test_indices, first_checkpoint, second_checkpoint,
                     artifacts, *, device="cpu", n_puzzles=256, attempts=32, steps=100,
                     batch_size=128, seed=17, bootstrap_draws=3000, refresh=False,
                     progress=None):
    """Same held-out puzzles and common sampling seeds; checkpoint-specific caches."""
    if attempts < 1 or steps < 1 or batch_size < 1 or bootstrap_draws < 1:
        raise ValueError("Evaluation counts must be positive.")
    if n_puzzles is not None and n_puzzles < 1:
        raise ValueError("Use a positive puzzle count, or None for all shared test puzzles.")
    first, first_info = load_queens_dfm(first_checkpoint, dataset_sha256=puzzles["metadata"]["data_sha256"], device=device)
    second, second_info = load_queens_dfm(second_checkpoint, dataset_sha256=puzzles["metadata"]["data_sha256"], device=device)
    if (first_info["conditioning_mode"] != "coloring_each_step"
            or second_info["conditioning_mode"] != "initial_state_only"):
        raise ValueError("Compare the coloring-conditioned reference with the initial-state-only checkpoint.")
    if min(first_info["global_step"], second_info["global_step"]) < 1:
        raise ValueError("Train both models before evaluating them.")
    if first_info["checkpoint_sha256"] == second_info["checkpoint_sha256"]:
        raise ValueError("Choose two distinct trained checkpoints.")
    common = first_info["test_indices"][
        np.isin(first_info["test_indices"], second_info["test_indices"])
        & np.isin(first_info["test_indices"], test_indices)
    ]
    if not len(common):
        raise ValueError("No common held-out puzzles.")
    indices = np.random.default_rng(seed + 10_000).permutation(common)
    if n_puzzles is not None:
        indices = indices[:n_puzzles]
    artifacts = Path(artifacts)
    successful, origins = [], []
    for label, candidate, info, filename, fallbacks in [
        ("DFM + coloring", first, first_info, "queens7_dfm_conditioning_reference_evaluation.npz",
         [artifacts / "queens7_dfm_vs_d3pm_dfm_evaluation.npz"]),
        ("DFM initial only", second, second_info, "queens7_dfm_conditioning_initial_only_evaluation.npz", []),
    ]:
        outcomes, origin = _evaluate(
            candidate, info, puzzles, indices, steps=steps, attempts=attempts,
            seed=seed + 40_000, batch_size=batch_size, cache_path=artifacts / filename,
            fallback_paths=fallbacks, refresh=refresh, progress=progress, label=label,
        )
        successful.append(outcomes)
        origins.append(origin)
    # One shared puzzle bootstrap for both runs preserves the paired comparison.
    n = len(indices)
    weights = np.random.default_rng(seed + 40_001).multinomial(
        n, np.full(n, 1 / n), size=bootstrap_draws,
    )
    models = []
    boot_curves = []
    for label, outcomes, info, origin in zip(["DFM + coloring", "DFM initial only"], successful,
                                           [first_info, second_info], origins):
        per_puzzle = _pass_at_k(outcomes)
        boot = weights @ per_puzzle / n
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        probability = outcomes.mean(axis=1)
        boot_curves.append(boot)
        models.append(dict(
            label=label, seed=info["seed"], epochs=info["epochs"],
            optimizer_steps=info["global_step"], checkpoint_sha256=info["checkpoint_sha256"],
            conditioning_mode=info["conditioning_mode"], parameters=info["parameters"],
            checkpoint_path=info["checkpoint_path"], evaluation_origin=origin,
            counts=outcomes.sum(axis=1).tolist(), probability=probability.tolist(),
            mean_probability=float(probability.mean()), pass_at_k=per_puzzle.mean(axis=0).tolist(),
            lower=lower.tolist(), upper=upper.tolist(),
            from_mean=(1 - (1 - probability.mean())**np.arange(attempts + 1)).tolist(),
        ))
    x, y = [np.asarray(model["probability"]) for model in models]
    variable = len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0
    difference = boot_curves[1] - boot_curves[0]
    return dict(
        dataset_indices=indices.tolist(), attempts=attempts, steps=steps,
        attempt_counts=list(range(attempts + 1)), models=models,
        dataset_sha256=puzzles["metadata"]["data_sha256"],
        pearson=float(np.corrcoef(x, y)[0, 1]) if variable else None,
        spearman=float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]) if variable else None,
        mean_probability_difference=float(y.mean() - x.mean()),
        mean_difference_interval=np.quantile(difference[:, 1], [0.025, 0.975]).tolist(),
        pass_at_k_difference_lower=np.quantile(difference, 0.025, axis=0).tolist(),
        pass_at_k_difference_upper=np.quantile(difference, 0.975, axis=0).tolist(),
    )
