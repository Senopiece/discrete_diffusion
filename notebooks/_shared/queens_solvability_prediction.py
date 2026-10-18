"""Deduction features, information diagnostics, and grouped solvability prediction."""

from collections import Counter
import hashlib
from itertools import groupby
import re

import numpy as np

from .queens_deduction_metrics import queen_dependency_depth


def _entropy(values):
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def trace_feature_vector(coloring, deductions):
    """Return finite numeric features without outcomes, source IDs, or absolute color IDs."""
    steps = [
        dict(state=d["state"], cells=d["cells"], reason=d["reason"])
        if isinstance(d, dict) else dict(state=d.state, cells=d.cells, reason=d.reason)
        for d in deductions
    ]
    graph = queen_dependency_depth(coloring, steps)
    size = len(coloring)
    crosses = [d for d in steps if d["state"] == "crossedOut"]
    stars = [d for d in steps if d["state"] == "star"]
    waits = [q["preceding_elimination_steps"] for q in graph["queens"]]
    cross_sizes = [len(d["cells"]) for d in crosses]

    def kind(d):
        reason = d["reason"]
        if reason in ("This position cannot be a star", "This position must be a star"):
            return "lookahead"
        if reason == "X-Wing deduction":
            return "xwing"
        if "can only have" in reason or " must have " in reason:
            return "subset"
        if d["state"] == "star":
            return "forced_queen"
        return "local_contradiction"

    kinds = [kind(d) for d in steps]
    counts = Counter(kinds)
    subsets = [d for d in steps if kind(d) == "subset"]
    subset_zones = [len(re.findall(r"\{zone_\d+\}", d["reason"])) for d in subsets]
    runs = [(state, len(list(items))) for state, items in groupby(d["state"] for d in steps)]
    queen_runs = [count for state, count in runs if state == "star"]
    cross_runs = [count for state, count in runs if state == "crossedOut"]
    n_cross = len(crosses)
    features = dict(
        trace_length=len(steps),
        elimination_steps=n_cross,
        explicitly_excluded_cells=sum(cross_sizes),
        singleton_eliminations=sum(value == 1 for value in cross_sizes),
        batch_eliminations=sum(value > 1 for value in cross_sizes),
        mean_elimination_batch=float(np.mean(cross_sizes)) if crosses else 0.0,
        max_elimination_batch=max(cross_sizes, default=0),
        subset_steps=counts["subset"],
        multi_zone_subset_steps=sum(value > 1 for value in subset_zones),
        mean_subset_zone_count=float(np.mean(subset_zones)) if subsets else 0.0,
        max_subset_zone_count=max(subset_zones, default=0),
        local_contradiction_steps=counts["local_contradiction"],
        lookahead_steps=counts["lookahead"],
        xwing_steps=counts["xwing"],
        queens_forced_by_region=sum("Last option in {zone_" in d["reason"] for d in stars),
        queens_forced_by_row=sum("Last option in row" in d["reason"] for d in stars),
        queens_forced_by_column=sum("Last option in column" in d["reason"] for d in stars),
        mean_wait=float(np.mean(waits)),
        std_wait=float(np.std(waits)),
        first_queen_wait=waits[0],
        mean_early_wait=float(np.mean(waits[:max(1, size // 2)])),
        mean_late_wait=float(np.mean(waits[max(1, size // 2):])),
        normalized_mean_wait=float(np.mean(waits)) / max(1, n_cross),
        queen_run_count=len(queen_runs),
        longest_queen_run=max(queen_runs, default=0),
        longest_elimination_run=max(cross_runs, default=0),
        state_transitions=max(0, len(runs) - 1),
        rule_transitions=sum(a != b for a, b in zip(kinds[:-1], kinds[1:])),
        rule_entropy=_entropy(kinds),
        max_depth=graph["max_depth"],
        mean_depth=graph["mean_depth"],
        std_depth=float(np.std(graph["queen_depths"])),
        dependency_roots=graph["root_count"],
        dependency_edges=len(graph["edges"]),
        opaque_seeds=graph["opaque_seed_count"],
        max_parent_count=max(len(q["parents"]) for q in graph["queens"]),
        widest_dependency_layer=max(Counter(graph["queen_depths"]).values()),
    )
    cumulative_queens = np.cumsum([
        len(d["cells"]) if d["state"] == "star" else 0 for d in steps
    ])
    for quarter in (1, 2, 3):
        index = max(0, int(np.ceil(len(steps) * quarter / 4)) - 1)
        features[f"queens_by_trace_quarter_{quarter}"] = int(cumulative_queens[index])
    for i, value in enumerate(waits, start=1):
        features[f"queen_{i}_wait"] = value
    row_spans, col_spans = [], []
    for deduction in crosses:
        cells = np.asarray(deduction["cells"])
        row_spans.append(int(np.ptp(cells[:, 0])) if len(cells) else 0)
        col_spans.append(int(np.ptp(cells[:, 1])) if len(cells) else 0)
    features["mean_exclusion_row_span"] = float(np.mean(row_spans)) if crosses else 0.0
    features["mean_exclusion_column_span"] = float(np.mean(col_spans)) if crosses else 0.0
    features["single_row_exclusions"] = sum(span == 0 for span in row_spans)
    features["single_column_exclusions"] = sum(span == 0 for span in col_spans)
    return {name: float(value) for name, value in features.items()}


def solution_group(deductions, size):
    """Group identical queen placements under all eight square symmetries."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for step in deductions:
        state = step["state"] if isinstance(step, dict) else step.state
        if state == "star":
            cells = step["cells"] if isinstance(step, dict) else step.cells
            for row, col in cells:
                mask[row, col] = 1
    canonical = min(
        array.tobytes() for turns in range(4)
        for array in (np.rot90(mask, turns), np.fliplr(np.rot90(mask, turns)))
    )
    return hashlib.sha256(canonical).hexdigest()


def build_feature_dataset(puzzles, evaluation, trace_records):
    if (
        "dataset_sha256" in evaluation
        and str(evaluation["dataset_sha256"]) != puzzles["metadata"]["data_sha256"]
    ):
        raise ValueError("Evaluation and dataset do not match.")
    indices = np.asarray(evaluation["dataset_indices"], dtype=np.int64)
    successes = np.asarray(evaluation["successes"], dtype=bool)
    if (
        indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices)
        or successes.ndim != 2 or len(successes) != len(indices) or not successes.shape[1]
        or np.any(indices < 0) or np.any(indices >= len(puzzles["regions"]))
    ):
        raise ValueError("Expected distinct dataset indices and aligned DFM attempts.")
    lookup = {row["fingerprint"]: row for row in trace_records}
    rows, kept, excluded, groups = [], [], [], []
    for i, index in enumerate(indices):
        trace = lookup.get(str(puzzles["fingerprints"][index]))
        if trace is None or "deduction_steps" not in trace:
            raise ValueError(f"Missing deduction trace for puzzle {index}.")
        if not trace["is_solved_deductively"]:
            excluded.append(int(index))
            continue
        board = puzzles["regions"][index]
        rows.append(trace_feature_vector(board, trace["deduction_steps"]))
        groups.append(solution_group(trace["deduction_steps"], len(board)))
        kept.append(i)
    if not rows:
        raise ValueError("No completed traces are available.")
    feature_names = list(rows[0])
    X = np.asarray([[row[name] for name in feature_names] for row in rows])
    if not np.isfinite(X).all():
        raise ValueError("Feature extraction produced nonfinite values.")
    selected_indices = indices[kept]
    return dict(
        X=X, y=successes[kept].mean(axis=1), successes=successes[kept],
        feature_names=feature_names, groups=np.asarray(groups),
        dataset_indices=selected_indices, excluded_indices=np.asarray(excluded),
        source_names=np.asarray(puzzles["source_names"])[selected_indices],
        source_ids=np.asarray(puzzles["source_ids"])[selected_indices],
    )


def _quantile_codes(values, max_bins=4):
    unique = np.unique(values)
    if len(unique) <= max_bins:
        return np.searchsorted(unique, values), len(unique)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, max_bins + 1)))
    codes = np.searchsorted(edges[1:-1], values, side="right")
    return codes, int(codes.max()) + 1


def _table_mi(tables):
    probability = tables / tables.sum(axis=(-2, -1), keepdims=True)
    independent = probability.sum(axis=-1, keepdims=True) * probability.sum(
        axis=-2, keepdims=True
    )
    term = np.zeros_like(probability, dtype=float)
    selected = probability > 0
    term[selected] = probability[selected] * np.log2(
        probability[selected] / np.broadcast_to(independent, probability.shape)[selected]
    )
    return term.sum(axis=(-2, -1))


def feature_information(X, y, feature_names, *, permutations=499, seed=17):
    """Descriptive I(binned X_j; binned measured solvability), in bits."""
    if permutations < 1:
        raise ValueError("Use at least one permutation.")
    target = np.searchsorted([0.25, 0.5, 0.75], y, side="right")
    rng = np.random.default_rng(seed)
    shuffled = np.stack([rng.permutation(target) for _ in range(permutations)])
    results = []
    for column, name in enumerate(feature_names):
        codes, count = _quantile_codes(X[:, column])
        table = np.bincount(codes * 4 + target, minlength=count * 4).reshape(count, 4)
        stride = count * 4
        perm_codes = (
            codes[None, :] * 4 + shuffled
            + np.arange(permutations)[:, None] * stride
        )
        perm_tables = np.bincount(
            perm_codes.ravel(), minlength=permutations * stride
        ).reshape(permutations, count, 4)
        value = float(_table_mi(table))
        null = _table_mi(perm_tables)
        results.append(dict(
            feature=name, mi_bits=value, null_mean_bits=float(null.mean()),
            excess_bits=value - float(null.mean()),
            null_95_bits=float(np.quantile(null, 0.95)), feature_bins=count,
        ))
    return dict(
        target_bins=target, entropy_bits=_entropy(target),
        features=sorted(results, key=lambda row: row["excess_bits"], reverse=True),
        permutations=permutations,
    )


def _group_bootstrap(groups, *, draws, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=draws
    ).astype(float)
    counts = np.bincount(inverse)
    denominators = weights @ counts
    return inverse, weights, denominators


def _bootstrap_means(values, inverse, weights, denominators):
    sums = np.bincount(inverse, weights=np.asarray(values), minlength=weights.shape[1])
    return weights @ sums / denominators


def _spearman(x, y):
    def ranks(v):
        _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - counts + (counts - 1) / 2)[inverse]
    a, b = ranks(x), ranks(y)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None


def fit_solvability_models(data, *, seed=17, outer_splits=5, inner_splits=3,
                          bootstrap_draws=2000, progress=None):
    """Nested grouped CV, probability-bin information, then refits for future inference."""
    from sklearn.base import clone
    from sklearn.ensemble import (
        HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.tree import DecisionTreeRegressor, export_text
    from threadpoolctl import threadpool_limits

    X, y, groups = data["X"], data["y"], data["groups"]
    n_groups = len(np.unique(groups))
    if n_groups < 4:
        raise ValueError("At least four distinct queen-placement groups are needed.")
    outer_count = min(outer_splits, n_groups)
    outer = list(GroupKFold(
        n_splits=outer_count, shuffle=True, random_state=seed
    ).split(X, y, groups))
    specifications = {
        "tree": (
            DecisionTreeRegressor(random_state=seed),
            {"max_depth": [2, 3, 5], "min_samples_leaf": [8, 16]},
        ),
        "forest": (
            RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=seed),
            {"max_depth": [5, None], "min_samples_leaf": [4, 8, 16], "max_features": [0.7]},
        ),
        "boosting": (
            HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.05, min_samples_leaf=12,
                early_stopping=False, random_state=seed,
            ),
            {"max_leaf_nodes": [3, 7], "l2_regularization": [1.0, 10.0]},
        ),
    }
    class_model = RandomForestClassifier(
        n_estimators=200, max_features=0.7, n_jobs=1, random_state=seed,
    )
    class_grid = {"max_depth": [5, None], "min_samples_leaf": [8, 16]}
    target_bins = np.searchsorted([0.25, 0.5, 0.75], y, side="right")
    predictions = {name: np.empty(len(y)) for name in ["mean", *specifications]}
    probabilities = np.empty((len(y), 4))
    fold_id = np.empty(len(y), dtype=int)
    fold_parameters = []
    tasks = outer_count * (len(specifications) + 1) + len(specifications) + 1
    completed = 0

    def update(stage, increment=False):
        nonlocal completed
        completed += int(increment)
        if progress:
            progress(stage, completed, tasks)

    def inner_cv(train):
        count = min(inner_splits, len(np.unique(groups[train])))
        return list(GroupKFold(n_splits=count).split(X[train], y[train], groups[train]))

    def class_score(estimator, features, labels):
        raw = estimator.predict_proba(features)
        q = np.full((len(labels), 4), 1e-6)
        q[:, estimator.classes_.astype(int)] = np.maximum(raw, 1e-6)
        q /= q.sum(axis=1, keepdims=True)
        return float(np.log(q[np.arange(len(labels)), labels]).mean())

    with threadpool_limits(limits=1):
        for fold, (train, test) in enumerate(outer):
            if set(groups[train]) & set(groups[test]):
                raise RuntimeError("A queen-placement group leaked across an outer fold.")
            cv = inner_cv(train)
            fold_id[test] = fold
            predictions["mean"][test] = y[train].mean()
            for name, (estimator, grid) in specifications.items():
                update(f"Outer fold {fold + 1}/{outer_count}: {name}")
                search = GridSearchCV(
                    clone(estimator), grid, cv=cv, scoring="neg_mean_squared_error",
                    n_jobs=1, error_score="raise",
                ).fit(X[train], y[train])
                predictions[name][test] = np.clip(search.predict(X[test]), 0, 1)
                fold_parameters.append(dict(
                    fold=fold, model=name, parameters=search.best_params_,
                    inner_mse=-float(search.best_score_),
                    test_dataset_indices=data["dataset_indices"][test].tolist(),
                ))
                update(name, increment=True)
            update(f"Outer fold {fold + 1}/{outer_count}: joint-information classifier")
            class_search = GridSearchCV(
                clone(class_model), class_grid, cv=cv, scoring=class_score,
                n_jobs=1, error_score="raise",
            ).fit(X[train], target_bins[train])
            raw = class_search.predict_proba(X[test])
            q = np.full((len(test), 4), 1e-6)
            q[:, class_search.best_estimator_.classes_.astype(int)] = np.maximum(raw, 1e-6)
            probabilities[test] = q / q.sum(axis=1, keepdims=True)
            update("joint information", increment=True)

        models, final_parameters = {}, {}
        full_train = np.arange(len(y))
        final_cv = inner_cv(full_train)
        for name, (estimator, grid) in specifications.items():
            update(f"Final refit: {name}")
            search = GridSearchCV(
                clone(estimator), grid, cv=final_cv, scoring="neg_mean_squared_error",
                n_jobs=1, error_score="raise",
            ).fit(X, y)
            models[name] = search.best_estimator_
            final_parameters[name] = search.best_params_
            update(name, increment=True)
        update("Final refit: probability-bin classifier")
        search = GridSearchCV(
            clone(class_model), class_grid, cv=final_cv, scoring=class_score,
            n_jobs=1, error_score="raise",
        ).fit(X, target_bins)
        models["information_classifier"] = search.best_estimator_
        final_parameters["information_classifier"] = search.best_params_
        update("complete", increment=True)

    inverse, weights, denominators = _group_bootstrap(
        groups, draws=bootstrap_draws, seed=seed + 1
    )
    mean_y = _bootstrap_means(y, inverse, weights, denominators)
    variance_boot = _bootstrap_means(y * y, inverse, weights, denominators) - mean_y**2
    variance = float(np.var(y))
    metrics = []
    base_error = (predictions["mean"] - y)**2
    for name, prediction in predictions.items():
        squared, absolute = (prediction - y)**2, np.abs(prediction - y)
        mse = float(squared.mean())
        mse_boot = _bootstrap_means(squared, inverse, weights, denominators)
        r2_boot = 1 - mse_boot / np.maximum(variance_boot, 1e-12)
        q = np.clip(prediction, 1e-6, 1 - 1e-6)
        cross_entropy = -(y * np.log2(q) + (1 - y) * np.log2(1 - q))
        interval = np.quantile(100 * np.sqrt(mse_boot), [0.025, 0.975])
        gain = _bootstrap_means(base_error - squared, inverse, weights, denominators)
        metrics.append(dict(
            model=name, mse=mse, rmse_pp=100 * np.sqrt(mse),
            rmse_lower_pp=float(interval[0]), rmse_upper_pp=float(interval[1]),
            mae_pp=100 * float(absolute.mean()), r2=1 - mse / variance,
            r2_interval=np.quantile(r2_boot, [0.025, 0.975]).tolist(),
            spearman=_spearman(prediction, y),
            single_attempt_cross_entropy_bits=float(cross_entropy.mean()),
            mse_improvement_over_baseline=float((base_error - squared).mean()),
            improvement_interval=np.quantile(gain, [0.025, 0.975]).tolist(),
            training_r2=None if name == "mean" else 1 - float(np.mean(
                (np.clip(models[name].predict(X), 0, 1) - y)**2
            )) / variance,
        ))
    entropy = _entropy(target_bins)
    log_probability = np.log2(probabilities[np.arange(len(y)), target_bins])
    lower_bound = entropy + float(log_probability.mean())
    entropy_draws = np.zeros(len(denominators))
    for category in range(4):
        marginal = _bootstrap_means(
            (target_bins == category).astype(float), inverse, weights, denominators
        )
        positive = marginal > 0
        entropy_draws[positive] -= marginal[positive] * np.log2(marginal[positive])
    info_draws = entropy_draws + _bootstrap_means(
        log_probability, inverse, weights, denominators
    )
    return dict(
        predictions=predictions, fold_id=fold_id, fold_parameters=fold_parameters,
        metrics=metrics, models=models, final_parameters=final_parameters,
        information=dict(
            target_bins=target_bins, oof_bin_probabilities=probabilities,
            entropy_bits=entropy, oof_cross_entropy_bits=-float(log_probability.mean()),
            joint_lower_bound_bits=lower_bound,
            bootstrap_interval_bits=np.quantile(info_draws, [0.025, 0.975]).tolist(),
        ),
        tree_text=export_text(
            models["tree"], feature_names=data["feature_names"], max_depth=5, decimals=3
        ),
        group_count=n_groups,
    )


def source_information_audit(data, fitted, *, bootstrap_draws=2000, seed=17):
    """Audit source confounding without adding source labels to the predictors."""
    y, sources = data["y"], data["source_names"]
    folds = fitted["fold_id"]
    prediction = np.empty(len(y))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        for source in np.unique(sources[test]):
            available = train & (sources == source)
            prediction[test & (sources == source)] = (
                y[available].mean() if available.any() else y[train].mean()
            )
    inverse, weights, denominators = _group_bootstrap(
        data["groups"], draws=bootstrap_draws, seed=seed
    )
    error = (prediction - y)**2
    mse = float(error.mean())
    errors_boot = _bootstrap_means(error, inverse, weights, denominators)
    interval = np.quantile(100 * np.sqrt(errors_boot), [0.025, 0.975])
    baseline = dict(
        model="source_mean", mse=mse, r2=1 - mse / float(np.var(y)),
        rmse_pp=100 * np.sqrt(mse), rmse_lower_pp=float(interval[0]),
        rmse_upper_pp=float(interval[1]),
        mae_pp=100 * float(np.abs(prediction - y).mean()),
        spearman=_spearman(prediction, y),
    )
    bins = fitted["information"]["target_bins"]
    conditional_entropy = sum(
        float(np.mean(sources == source)) * _entropy(bins[sources == source])
        for source in np.unique(sources)
    )
    q = fitted["information"]["oof_bin_probabilities"]
    logq = np.log2(q[np.arange(len(y)), bins])
    conditional_draws = np.zeros(bootstrap_draws)
    for source in np.unique(sources):
        selected = sources == source
        source_probability = _bootstrap_means(
            selected.astype(float), inverse, weights, denominators
        )
        for category in range(4):
            joint_probability = _bootstrap_means(
                (selected & (bins == category)).astype(float), inverse, weights, denominators
            )
            positive = joint_probability > 0
            conditional_draws[positive] -= joint_probability[positive] * np.log2(
                joint_probability[positive] / source_probability[positive]
            )
    bound_draws = conditional_draws + _bootstrap_means(logq, inverse, weights, denominators)
    gains = {}
    for name in ("tree", "forest", "boosting"):
        gain = error - (fitted["predictions"][name] - y)**2
        draws = _bootstrap_means(gain, inverse, weights, denominators)
        gains[name] = dict(
            mean_mse_improvement=float(gain.mean()),
            interval=np.quantile(draws, [0.025, 0.975]).tolist(),
        )
    return dict(
        source_mean_predictions=prediction, source_baseline=baseline,
        conditional_entropy_bits=float(conditional_entropy),
        source_information_bits=fitted["information"]["entropy_bits"] - conditional_entropy,
        conditional_lower_bound_bits=conditional_entropy + float(logq.mean()),
        conditional_lower_bound_interval=np.quantile(
            bound_draws, [0.025, 0.975]
        ).tolist(),
        improvement_over_source_baseline=gains,
    )
