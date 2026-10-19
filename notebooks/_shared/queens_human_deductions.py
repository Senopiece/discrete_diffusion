"""Deduction-trace features versus measured human time and difficulty."""

import json
import logging
import os
from pathlib import Path
import tempfile

import numpy as np

from .queens_data import EMPTY, solution_checks
from .queens_human_analysis import _human_records_by_puzzle, _spearman
from .queens_solvability_prediction import trace_feature_vector


SOLVER_COMMIT = "74bdd520eff149384afa3dd4c055b44e64ce0394"
SOLVER_MODE = "generate_deductions_v1"


def _read_cache(path, dataset_sha):
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (
        saved.get("dataset_sha256") == dataset_sha
        and saved.get("solver_commit") == SOLVER_COMMIT
        and saved.get("solver_mode") == SOLVER_MODE
    ):
        return saved.get("puzzles", {})
    return {}


def _solve_deductively(board):
    from speed_stars.core import (
        EMPTY_CELL_VALUE, STAR_CELL_VALUE, cross_out_if_empty, is_solved, place_star,
    )
    from speed_stars.deductions import generate_deductions

    zones = board.tolist()
    size = len(board)
    partial = [[EMPTY_CELL_VALUE] * size for _ in range(size)]
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        deductions = generate_deductions(zones)
    finally:
        logging.disable(previous_logging_disable)
    for deduction in deductions:
        for cell in deduction.cells:
            if deduction.state == "star":
                partial = place_star(partial, zones, cell.x, cell.y)
            elif deduction.state == "crossedOut":
                cross_out_if_empty(partial, cell.x, cell.y)
            else:
                raise ValueError(f"Unknown deduction state: {deduction.state}")
    solved = bool(is_solved(partial, zones))
    if solved:
        tokens = np.where(np.asarray(partial) == STAR_CELL_VALUE, board, EMPTY)
        if not bool(solution_checks(board, tokens)["solved"]):
            raise RuntimeError("The deduction solver returned an invalid placement.")
    return dict(
        is_solved_deductively=solved, status="solved" if solved else "stalled",
        deduction_steps=[dict(
            state=item.state, cells=[[cell.x, cell.y] for cell in item.cells],
            reason=item.reason,
        ) for item in deductions],
        partial_grid=partial,
        queens_placed=int(np.sum(np.asarray(partial) == STAR_CELL_VALUE)),
        undecided_cells=int(np.sum(np.asarray(partial) == EMPTY_CELL_VALUE)),
    )


def human_deduction_study(puzzles, test_indices, csv_path, cache_path, *, progress=None):
    """All human test puzzles; compute missing deduction traces without model calls."""
    by_puzzle, counts = _human_records_by_puzzle(puzzles, test_indices, csv_path)
    cache_path = Path(cache_path)
    dataset_sha = puzzles["metadata"]["data_sha256"]
    cache = _read_cache(cache_path, dataset_sha)
    rows, stalled, cache_hits, generated = [], [], 0, 0
    feature_names = []
    for done, (index, plays) in enumerate(sorted(by_puzzle.items()), start=1):
        fingerprint = str(puzzles["fingerprints"][index])
        verdict = cache.get(fingerprint)
        if verdict is not None and "deduction_steps" in verdict:
            cache_hits += 1
        else:
            verdict = _solve_deductively(puzzles["regions"][index])
            cache[fingerprint] = verdict
            generated += 1
        if verdict["is_solved_deductively"]:
            features = trace_feature_vector(
                puzzles["regions"][index], verdict["deduction_steps"],
            )
            if not all(np.isfinite(value) for value in features.values()):
                raise ValueError(f"Nonfinite trace feature for puzzle {index}.")
            if not feature_names:
                feature_names = list(features)
            rows.append(dict(
                dataset_index=index, puzzle_id=fingerprint,
                source=str(puzzles["source_names"][index]),
                source_id=str(puzzles["source_ids"][index]), players=len(plays),
                median_seconds=float(np.median([row["seconds"] for row in plays])),
                median_difficulty=float(np.median([row["difficulty"] for row in plays])),
                **features,
            ))
        else:
            stalled.append(index)
        if progress:
            progress(done, len(by_puzzle), cache_hits)
    if generated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve other puzzles already cached by either notebook.
        merged = _read_cache(cache_path, dataset_sha)
        merged.update(cache)
        descriptor, temporary = tempfile.mkstemp(dir=cache_path.parent, suffix=".json.tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(dict(
                    dataset_sha256=dataset_sha, solver_commit=SOLVER_COMMIT,
                    solver_mode=SOLVER_MODE, puzzles=merged,
                ), output, ensure_ascii=False, indent=2)
                output.write("\n")
            os.replace(temporary, cache_path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    correlations = []
    for feature in feature_names:
        x = [row[feature] for row in rows]
        correlation = dict(
            feature=feature, distinct_values=len(set(x)),
            difficulty=_spearman(x, [row["median_difficulty"] for row in rows]),
            time=_spearman(x, [row["median_seconds"] for row in rows]),
            by_source={},
        )
        for source in sorted({row["source"] for row in rows}):
            subset = [row for row in rows if row["source"] == source]
            correlation["by_source"][source] = dict(
                n=len(subset),
                difficulty=_spearman([row[feature] for row in subset],
                                     [row["median_difficulty"] for row in subset]),
                time=_spearman([row[feature] for row in subset],
                               [row["median_seconds"] for row in subset]),
            )
        correlations.append(correlation)
    correlations.sort(
        key=lambda row: max(abs(row["difficulty"] or 0), abs(row["time"] or 0)),
        reverse=True,
    )
    return dict(
        rows=rows, feature_names=feature_names, correlations=correlations,
        matched_puzzles=len(rows), stalled_dataset_indices=stalled,
        cache_hits=cache_hits, newly_solved_puzzles=generated,
        dataset_sha256=dataset_sha, solver_commit=SOLVER_COMMIT, solver_mode=SOLVER_MODE,
        **counts,
    )
