"""Prepare synthetic, uniquely solvable 8x8 Queens pairs OFFLINE.

The notebook only loads the finished dataset. No external sources are used.
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
import time

import numpy as np
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
from _shared.queens_data import normalize_regions, puzzle_fingerprint

SIZE = 8
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MIN_REGION_CELLS = 3
MAX_REGION_CELLS = 14

@njit(cache=True)
def grow_regions(columns, rng):
    """One cardinal-neighbor cell per unblocked region per shuffled round."""
    regions = np.full((SIZE * SIZE), -1, dtype=np.int8)
    frontier = np.zeros((SIZE, (SIZE * SIZE)), dtype=np.bool_)
    for row in range(SIZE):
        regions[row * SIZE + columns[row]] = row
    for label in range(SIZE):
        row, col = label, columns[label]
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE and regions[nr * SIZE + nc] == -1:
                frontier[label, nr * SIZE + nc] = True
    free = (SIZE * (SIZE - 1))
    choices = np.empty((SIZE * SIZE), dtype=np.int64)
    while free:
        order = np.arange(SIZE)
        # Fisher-Yates: every region gets at most one turn in this round.
        for index in range(SIZE - 1, 0, -1):
            other = rng.integers(0, index + 1)
            order[index], order[other] = order[other], order[index]
        progressed = False
        for label in order:
            count = 0
            for cell in range((SIZE * SIZE)):
                if frontier[label, cell]:
                    choices[count] = cell
                    count += 1
            if not count:
                continue
            cell = choices[rng.integers(0, count)]
            regions[cell] = label
            for region in range(SIZE):
                frontier[region, cell] = False
            row, col = cell // SIZE, cell % SIZE
            for dr, dc in DIRECTIONS:
                nr, nc = row + dr, col + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE and regions[nr * SIZE + nc] == -1:
                    frontier[label, nr * SIZE + nc] = True
            free -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("Region growth stopped before the grid was filled.")
    return regions.reshape((SIZE, SIZE))


@njit(cache=True)
def unique_solution(board, candidates):
    """Exhaust all admissible permutations; stop early at the second solution."""
    count = 0
    for candidate in candidates:
        seen = 0
        valid = True
        for row in range(SIZE):
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
        for row in range(SIZE):
            bit = 1 << board[row, candidate[row]]
            if seen & bit:
                break
            seen |= bit
        else:
            count += 1
    return count


@njit(cache=True)
def remains_connected(board, label, expected_size):
    visited = np.zeros((SIZE * SIZE), dtype=np.bool_)
    queue = np.empty((SIZE * SIZE), dtype=np.int64)
    tail = 0
    for cell in range((SIZE * SIZE)):
        if board[cell // SIZE, cell % SIZE] == label:
            queue[0] = cell
            visited[cell] = True
            tail = 1
            break
    head = 0
    while head < tail:
        cell = queue[head]
        head += 1
        row, col = cell // SIZE, cell % SIZE
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE:
                neighbor = nr * SIZE + nc
                if not visited[neighbor] and board[nr, nc] == label:
                    visited[neighbor] = True
                    queue[tail] = neighbor
                    tail += 1
    return tail == expected_size


@njit(cache=True)
def refine_regions(board, columns, candidates, rng, max_moves=6000):
    """Move boundary cells to eliminate alternatives while keeping region shape constraints."""
    sizes = np.zeros(SIZE, dtype=np.int64)
    for row in range(SIZE):
        for col in range(SIZE):
            sizes[board[row, col]] += 1
    current_count = solution_count(board, candidates)
    if current_count == 1:
        return board, 0
    for move in range(max_moves):
        cell = rng.integers(0, (SIZE * SIZE))
        row, col = cell // SIZE, cell % SIZE
        if columns[row] == col:
            continue
        old = board[row, col]
        if sizes[old] <= MIN_REGION_CELLS:
            continue
        dr, dc = DIRECTIONS[rng.integers(0, 4)]
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= SIZE or nc < 0 or nc >= SIZE:
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
    output = PROJECT_ROOT / "data" / "queens8"
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "puzzles.jsonl"
    manifest_path = output / "manifest.json"
    candidates = np.asarray(list(itertools.permutations(range(SIZE))), dtype=np.int64)
    candidates = candidates[np.all(np.abs(np.diff(candidates, axis=1)) != 1, axis=1)]
    rng = np.random.default_rng(seed)
    limit = max_attempts if max_attempts is not None else count * 1000
    records, fingerprints, rejected = [], set(), Counter()
    started = time.monotonic()
    print(f"Preparing {count} synthetic {SIZE}x{SIZE} puzzles; exact uniqueness checks.", flush=True)
    attempts = 0
    while len(records) < count and attempts < limit:
        attempts += 1
        columns = candidates[int(rng.integers(len(candidates)))]
        regions = grow_regions(columns, rng)
        sizes = np.bincount(regions.ravel(), minlength=SIZE)
        if sizes.min() < MIN_REGION_CELLS or sizes.max() > MAX_REGION_CELLS:
            rejected["region_size"] += 1
            continue
        regions, moves = refine_regions(regions, columns, candidates, rng)
        if moves < 0 or not unique_solution(regions, candidates):
            rejected["not_unique"] += 1
            continue
        board = normalize_regions(regions)
        fingerprint = puzzle_fingerprint(board)
        if fingerprint in fingerprints:
            rejected["symmetry_duplicate"] += 1
            continue
        fingerprints.add(fingerprint)
        records.append(dict(
            id=f"q8_{fingerprint[:16]}", fingerprint=fingerprint,
            regions=board.tolist(),
            queens=[[row, int(col)] for row, col in enumerate(columns)],
            unique_solution=True,
        ))
        if len(records) % 100 == 0 or len(records) == count:
            print(f"{len(records):,}/{count:,} puzzles | {attempts:,} candidates | "
                  f"{time.monotonic() - started:.1f}s", flush=True)
    if len(records) != count:
        raise RuntimeError(f"Only generated {len(records)}/{count}; existing dataset was not replaced.")
    records.sort(key=lambda record: record["id"])
    payload = "".join(json.dumps(record, separators=(",", ":")) + "\n"
                      for record in records).encode("utf-8")
    manifest = dict(
        version="queens8_synthetic_v1", board_size=SIZE, count=count, seed=seed,
        prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        data_sha256=hashlib.sha256(payload).hexdigest(),
        generator="scripts/generate_queens8.py", candidate_boards=attempts,
        region_cells=dict(min=MIN_REGION_CELLS, max=MAX_REGION_CELLS),
        rules="One queen per row, column and region; queens cannot touch diagonally.",
        growth="One adjacent cell per region per shuffled round; then connected boundary refinement.",
        uniqueness="Exactly one solution: all non-touching row/column permutations checked offline.",
        symmetry_deduplicated=True, rejected_candidates=dict(rejected),
    )
    data_path.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {data_path} | SHA256 {manifest['data_sha256']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()
    if args.count < 3:
        parser.error("--count must be at least 3")
    generate(args.count, args.seed, args.max_attempts)
