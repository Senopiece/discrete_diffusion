"""Optional OFFLINE synthetic addition to the ready-made Queens dataset.

Run once: uv run python scripts/generate_queens7.py --count 5000
The notebook only reads the resulting files. It never imports this generator.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
from _shared.queens_data import normalize_regions, puzzle_fingerprint

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MIN_REGION_CELLS = 2
MAX_REGION_CELLS = 12


@njit(cache=True)
def grow_regions(columns, rng):
    """One cardinal-neighbor cell per unblocked region per shuffled round."""
    regions = np.full(49, -1, dtype=np.int8)
    frontier = np.zeros((7, 49), dtype=np.bool_)
    for row in range(7):
        regions[row * 7 + columns[row]] = row
    for label in range(7):
        row, col = label, columns[label]
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 7 and 0 <= nc < 7 and regions[nr * 7 + nc] == -1:
                frontier[label, nr * 7 + nc] = True
    free = 42
    choices = np.empty(49, dtype=np.int64)
    while free:
        order = np.arange(7)
        # Fisher-Yates: every region gets at most one turn in this round.
        for index in range(6, 0, -1):
            other = rng.integers(0, index + 1)
            order[index], order[other] = order[other], order[index]
        progressed = False
        for label in order:
            count = 0
            for cell in range(49):
                if frontier[label, cell]:
                    choices[count] = cell
                    count += 1
            if not count:
                continue
            cell = choices[rng.integers(0, count)]
            regions[cell] = label
            for region in range(7):
                frontier[region, cell] = False
            row, col = cell // 7, cell % 7
            for dr, dc in DIRECTIONS:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 7 and 0 <= nc < 7 and regions[nr * 7 + nc] == -1:
                    frontier[label, nr * 7 + nc] = True
            free -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("Region growth stopped before the grid was filled.")
    return regions.reshape((7, 7))


@njit(cache=True)
def unique_solution(board, candidates):
    """Exhaust all admissible permutations; stop early at the second solution."""
    count = 0
    for candidate in candidates:
        seen = 0
        valid = True
        for row in range(7):
            bit = 1 << board[row, candidate[row]]
            if seen & bit:
                valid = False
                break
            seen |= bit
        if valid:
            count += 1
            if count > 1:
                return False
    return count == 1


@njit(cache=True)
def solution_count(board, candidates):
    count = 0
    for candidate in candidates:
        seen = 0
        for row in range(7):
            bit = 1 << board[row, candidate[row]]
            if seen & bit:
                break
            seen |= bit
        else:
            count += 1
    return count


@njit(cache=True)
def remains_connected(board, label, expected_size):
    visited = np.zeros(49, dtype=np.bool_)
    queue = np.empty(49, dtype=np.int64)
    tail = 0
    for cell in range(49):
        if board[cell // 7, cell % 7] == label:
            queue[0] = cell
            visited[cell] = True
            tail = 1
            break
    head = 0
    while head < tail:
        cell = queue[head]
        head += 1
        row, col = cell // 7, cell % 7
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 7 and 0 <= nc < 7:
                neighbor = nr * 7 + nc
                if not visited[neighbor] and board[nr, nc] == label:
                    visited[neighbor] = True
                    queue[tail] = neighbor
                    tail += 1
    return tail == expected_size


@njit(cache=True)
def refine_regions(board, columns, candidates, rng, max_moves=6000):
    """Move boundary cells to eliminate alternatives while keeping region shape constraints."""
    sizes = np.zeros(7, dtype=np.int64)
    for row in range(7):
        for col in range(7):
            sizes[board[row, col]] += 1
    current_count = solution_count(board, candidates)
    if current_count == 1:
        return board, 0
    for move in range(max_moves):
        cell = rng.integers(0, 49)
        row, col = cell // 7, cell % 7
        if columns[row] == col:
            continue
        old = board[row, col]
        if sizes[old] <= MIN_REGION_CELLS:
            continue
        dr, dc = DIRECTIONS[rng.integers(0, 4)]
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= 7 or nc < 0 or nc >= 7:
            continue
        new = board[nr, nc]
        if new == old or sizes[new] >= MAX_REGION_CELLS:
            continue
        board[row, col] = new
        if not remains_connected(board, old, sizes[old] - 1):
            board[row, col] = old
            continue
        proposed_count = solution_count(board, candidates)
        temperature = 0.1 + 3.0 * (1.0 - (move % 1000) / 1000.0) ** 2
        delta = proposed_count - current_count
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            sizes[old] -= 1
            sizes[new] += 1
            current_count = proposed_count
            if current_count == 1:
                return board, move + 1
        else:
            board[row, col] = old
    return board, -1


def generate(count=5000, seed=1701, max_attempts=None):
    output = PROJECT_ROOT / "data" / "queens7"
    manifest_path = output / "manifest.json"
    data_path = output / "puzzles.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError("First run scripts/collect_queens7.py.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    previous_synthetic = [record for record in records if record["sources"][0]["name"] == "Synthetic"]
    records = [record for record in records if record["sources"][0]["name"] != "Synthetic"]
    candidates = np.asarray(list(itertools.permutations(range(7))), dtype=np.int64)
    candidates = candidates[np.all(np.abs(np.diff(candidates, axis=1)) != 1, axis=1)]

    fingerprints = {record["fingerprint"] for record in records}
    rng = np.random.default_rng(seed)
    limit = max_attempts if max_attempts is not None else max(1000, count * 1000)
    added = 0
    rejected = Counter()
    print(
        f"Replacing {len(previous_synthetic)} synthetic puzzles with balanced growth and refinement "
        f"| region sizes {MIN_REGION_CELLS}..{MAX_REGION_CELLS}", flush=True,
    )
    attempts = 0
    while added < count and attempts < limit:
        attempts += 1
        if attempts % 50_000 == 0:
            print(f"Round-robin candidates: {attempts:,} | accepted: {added}/{count}", flush=True)
        columns = candidates[int(rng.integers(len(candidates)))]
        regions = grow_regions(columns, rng)
        sizes = np.bincount(regions.ravel(), minlength=7)
        if sizes.min() < MIN_REGION_CELLS:
            rejected["region_too_small"] += 1
            continue
        if sizes.max() > MAX_REGION_CELLS:
            rejected["region_too_large"] += 1
            continue
        regions, refinement_moves = refine_regions(regions, columns, candidates, rng)
        if refinement_moves < 0:
            rejected["refinement_not_unique"] += 1
            continue
        if not unique_solution(regions, candidates):
            rejected["not_unique"] += 1
            continue
        board = normalize_regions(regions)
        fingerprint = puzzle_fingerprint(board)
        if fingerprint in fingerprints:
            rejected["duplicate"] += 1
            continue
        fingerprints.add(fingerprint)
        records.append({
            "id": f"q7_{fingerprint[:16]}",
            "fingerprint": fingerprint,
            "regions": board.tolist(),
            "queens": [[row, int(col)] for row, col in enumerate(columns)],
            "unique_solution": "exhaustive_offline_7x7_check",
            "sources": [{
                "name": "Synthetic",
                "id": f"seed{seed}_attempt{attempts}",
                "url": "scripts/generate_queens7.py",
                "seed": seed,
                "attempt": attempts,
                "generator": "balanced_round_robin_then_uniqueness_refinement_v4",
                "min_region_cells": MIN_REGION_CELLS,
                "max_region_cells": MAX_REGION_CELLS,
                "refinement_moves": refinement_moves,
                "uniqueness": "exactly one admissible permutation, checked OFFLINE",
            }],
        })
        added += 1
        if added % 500 == 0:
            print(f"Synthetic pairs: {added}/{count} | candidate boards: {attempts}", flush=True)
    if added != count:
        raise RuntimeError(f"Collected {added}/{count} synthetic pairs in {attempts} attempts.")

    records.sort(key=lambda record: record["id"])
    payload = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n" for record in records
    ).encode("utf-8")
    manifest["version"] = "queens7_multisource_plus_synthetic_balanced_refined_v4"
    manifest["prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["synthetic"] = {
        "count": count, "seed": seed, "candidate_boards": attempts,
        "script": "scripts/generate_queens7.py",
        "checked_offline": True,
        "min_region_cells": MIN_REGION_CELLS,
        "max_region_cells": MAX_REGION_CELLS,
        "growth": "round robin: one random adjacent free cell per region per shuffled round; skip blocked regions",
        "refinement": "boundary-cell simulated annealing minimizes exact solution count; preserve queens, connectivity and region sizes",
        "previous_synthetic_count": len(previous_synthetic),
        "replaced_previous_synthetic": len(previous_synthetic),
        "retained": 0,
        "generated_new": count,
        "rejected_candidates": dict(rejected),
    }
    manifest["source_counts"] = dict(Counter(record["sources"][0]["name"] for record in records))
    manifest["counts"]["accepted"] = len(records)
    manifest["counts"]["synthetic"] = count
    manifest["uniqueness"] = (
        "external: source claims; synthetic: exhaustive offline 7x7 check; "
        "no searching during notebook execution"
    )
    manifest["data_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["notes"] = [
        "External solution coordinates were supplied by the original sources.",
        "External answer validity was checked; external uniqueness claims were trusted.",
        "Synthetic solutions were planted and uniqueness was checked exhaustively offline.",
        "Every synthetic region occupies at least two cells; no singleton queen regions.",
        "Synthetic boards start with round-robin growth, then boundary refinement preserves connectivity and 2..12 cells per region.",
        "AMAZE pairs originate in its published test split; our split is independent.",
        "Notebook execution only reads the prepared dataset and never generates or solves puzzles.",
    ]
    data_path.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(records), "sources": manifest["source_counts"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count cannot be negative")
    generate(args.count, args.seed, args.max_attempts)

