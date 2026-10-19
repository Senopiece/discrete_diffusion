"""Join human play records to per-puzzle Monte Carlo model success counts."""

from collections import defaultdict

import numpy as np
from scipy.stats import beta, rankdata

from .queens_human_game import read_human_games


def _spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or not np.ptp(x) or not np.ptp(y):
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _correlation(x, y, *, rng, bootstrap_draws):
    estimate = _spearman(x, y)
    if estimate is None:
        return dict(n=len(x), spearman=None, interval=None)
    draws = []
    for _ in range(bootstrap_draws):
        indices = rng.integers(0, len(x), len(x))
        value = _spearman(np.asarray(x)[indices], np.asarray(y)[indices])
        if value is not None:
            draws.append(value)
    return dict(
        n=len(x), spearman=estimate,
        interval=np.quantile(draws, [0.025, 0.975]).tolist() if len(draws) >= 20 else None,
    )


def _human_records_by_puzzle(puzzles, test_indices, csv_path):
    """Validate saved games and keep the first completion per player/puzzle."""
    test_set = set(np.asarray(test_indices, dtype=int).tolist())
    records = read_human_games(csv_path)
    usable, ignored = [], 0
    for row in records:
        try:
            index = int(row["dataset_index"])
            seconds, difficulty = float(row["elapsed_seconds"]), float(row["difficulty"])
            if (
                row["dataset_sha256"] != puzzles["metadata"]["data_sha256"]
                or index not in test_set
                or str(puzzles["fingerprints"][index]) != row["puzzle_id"]
                or not np.isfinite(seconds) or seconds <= 0
                or not np.isfinite(difficulty) or not 0 <= difficulty <= 10
            ):
                ignored += 1
                continue
            usable.append(dict(
                index=index, player=row["player_id"].strip() or "local",
                seconds=seconds, difficulty=difficulty, started=row["started_at_utc"],
            ))
        except (ValueError, KeyError, IndexError):
            ignored += 1
    first_plays = {}
    for row in sorted(usable, key=lambda row: row["started"]):
        first_plays.setdefault((row["player"], row["index"]), row)
    by_puzzle = defaultdict(list)
    for row in first_plays.values():
        by_puzzle[row["index"]].append(row)
    return by_puzzle, dict(
        recorded_games=len(records), usable_games=len(usable), ignored_games=ignored,
        first_player_puzzle_plays=len(first_plays),
        repeated_completed_plays_removed=len(usable) - len(first_plays),
        unique_human_puzzles=len(by_puzzle),
    )


def fit_human_time_curves(rows):
    """OLS rating = a + b*f(time in minutes), over the observed time range."""
    minutes = np.asarray([row["median_seconds"] / 60 for row in rows], dtype=float)
    ratings = np.asarray([row["median_difficulty"] for row in rows], dtype=float)
    if len(rows) < 2 or not np.ptp(minutes):
        return {}
    grid = np.linspace(minutes.min(), minutes.max(), 300)
    fits = {}
    for name, transform in [("log", np.log), ("sqrt", np.sqrt)]:
        design = np.column_stack([np.ones(len(minutes)), transform(minutes)])
        coefficients, _, rank, _ = np.linalg.lstsq(design, ratings, rcond=None)
        if rank < 2:
            continue
        prediction = design @ coefficients
        mse = float(np.mean((ratings - prediction)**2))
        variance = float(np.var(ratings))
        fits[name] = dict(
            intercept=float(coefficients[0]), slope=float(coefficients[1]),
            rmse=float(np.sqrt(mse)), r2=1 - mse / variance if variance > 0 else None,
            n=len(rows), time_unit="minutes", metrics="in_sample",
            curve_minutes=grid.tolist(),
            curve_difficulty=(coefficients[0] + coefficients[1] * transform(grid)).tolist(),
        )
    return fits


def human_time_difficulty_study(puzzles, test_indices, csv_path, *,
                                bootstrap_draws=2000, seed=17):
    """Human time vs rating on all recorded test puzzles, independent of models."""
    by_puzzle, counts = _human_records_by_puzzle(puzzles, test_indices, csv_path)
    rows = [dict(
        dataset_index=index, puzzle_id=str(puzzles["fingerprints"][index]),
        source=str(puzzles["source_names"][index]),
        source_id=str(puzzles["source_ids"][index]), players=len(plays),
        median_seconds=float(np.median([row["seconds"] for row in plays])),
        median_difficulty=float(np.median([row["difficulty"] for row in plays])),
    ) for index, plays in sorted(by_puzzle.items())]
    rng = np.random.default_rng(seed)
    correlations = {}
    for source, selected in [
        ("all", rows),
        *[(source, [row for row in rows if row["source"] == source])
          for source in sorted({row["source"] for row in rows})],
    ]:
        correlations[source] = _correlation(
            [row["median_seconds"] for row in selected],
            [row["median_difficulty"] for row in selected],
            rng=rng, bootstrap_draws=bootstrap_draws,
        )
    return dict(rows=rows, correlations=correlations, fits=fit_human_time_curves(rows), **counts)


def human_model_study(puzzles, test_indices, evaluation, csv_path, *,
                      prior_alpha=2.0, prior_beta=2.0, bootstrap_draws=2000, seed=17):
    """Use first completed play per person/puzzle, then one aggregate per puzzle.

    For p|c,A ~ Beta(c+a, A-c+b), return E[1/p|c,A] and a 95% credible
    interval for 1/p. Requiring a>1 keeps this mean finite for zero successes.
    The direct A/c estimate is retained separately, undefined for c=0.
    """
    if not (np.isfinite(prior_alpha) and prior_alpha > 1
            and np.isfinite(prior_beta) and prior_beta > 0):
        raise ValueError("Use prior_alpha > 1 and prior_beta > 0.")
    if (
        "dataset_sha256" in evaluation
        and str(evaluation["dataset_sha256"]) != puzzles["metadata"]["data_sha256"]
    ):
        raise ValueError("Evaluation and human-game dataset do not match.")
    indices = np.asarray(evaluation["dataset_indices"], dtype=int)
    successes = np.asarray(evaluation["successes"], dtype=bool)
    test_set = set(np.asarray(test_indices, dtype=int).tolist())
    if (
        indices.ndim != 1 or len(np.unique(indices)) != len(indices)
        or successes.ndim != 2 or len(successes) != len(indices)
        or successes.shape[1] < 1 or not set(indices.tolist()).issubset(test_set)
    ):
        raise ValueError("Expected aligned model attempts from the current test split.")
    by_puzzle, counts = _human_records_by_puzzle(puzzles, test_indices, csv_path)
    lookup = {int(index): position for position, index in enumerate(indices)}
    rows = []
    for index, plays in sorted(by_puzzle.items()):
        if index not in lookup:
            continue
        observed = successes[lookup[index]]
        count, attempts = int(observed.sum()), len(observed)
        alpha, beta_parameter = count + prior_alpha, attempts - count + prior_beta
        low_p, high_p = beta.ppf([0.025, 0.975], alpha, beta_parameter)
        rows.append(dict(
            dataset_index=index, puzzle_id=str(puzzles["fingerprints"][index]),
            source=str(puzzles["source_names"][index]),
            source_id=str(puzzles["source_ids"][index]),
            players=len(plays),
            median_seconds=float(np.median([row["seconds"] for row in plays])),
            median_difficulty=float(np.median([row["difficulty"] for row in plays])),
            model_successes=count, model_attempts=attempts,
            measured_probability=count / attempts,
            expected_attempts=(attempts + prior_alpha + prior_beta - 1)
                              / (count + prior_alpha - 1),
            expected_attempts_lower=1 / float(high_p),
            expected_attempts_upper=1 / float(low_p),
            direct_expected_attempts=attempts / count if count else None,
        ))
    rng = np.random.default_rng(seed)
    correlations = {}
    for name, selected in [
        ("all", rows),
        *[(str(source), [row for row in rows if row["source"] == source])
          for source in sorted({row["source"] for row in rows})],
    ]:
        x = np.asarray([row["expected_attempts"] for row in selected])
        correlations[name] = {
            "time": _correlation(
                x, [row["median_seconds"] for row in selected],
                rng=rng, bootstrap_draws=bootstrap_draws,
            ),
            "difficulty": _correlation(
                x, [row["median_difficulty"] for row in selected],
                rng=rng, bootstrap_draws=bootstrap_draws,
            ),
        }
    positive = [row for row in rows if row["model_successes"] > 0]
    direct_correlations = {
        name: _spearman(
            [row["direct_expected_attempts"] for row in positive],
            [row[field] for row in positive],
        )
        for name, field in [("time", "median_seconds"), ("difficulty", "median_difficulty")]
    }
    return dict(
        rows=rows, correlations=correlations,
        direct_positive_success_correlations=direct_correlations,
        prior_alpha=float(prior_alpha), prior_beta=float(prior_beta),
        **counts, matched_puzzles=len(rows),
        human_puzzles_without_model_evaluation=len(by_puzzle) - len(rows),
        evaluated_puzzles=len(indices),
    )
