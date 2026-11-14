"""Prepare synthetic, uniquely solvable 8x8 Queens pairs OFFLINE.

Generate connected regions and verify uniqueness by exhaustive enumeration.
"""

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
from _shared.queens_data import ARRAY_SHAPES, DATASET_VERSION, array_manifest

SIZE = 8
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MIN_REGION_CELLS = 3
MAX_REGION_CELLS = 14
ALL_REGIONS_ONCE = 0x11111111


def prepare_candidates():
    """Index every admissible row/column placement by the cells it uses."""
    candidates = np.asarray(list(itertools.permutations(range(SIZE))), dtype=np.int8)
    candidates = candidates[np.all(np.abs(np.diff(candidates, axis=1)) != 1, axis=1)]
    affected = []
    offsets = [0]
    for cell in range(SIZE * SIZE):
        affected.extend(np.flatnonzero(candidates[:, cell // SIZE] == cell % SIZE))
        offsets.append(len(affected))
    return candidates, np.asarray(affected, dtype=np.int32), np.asarray(offsets, dtype=np.int32)


@njit(cache=True, nogil=True)
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
def connected_mask(mask):
    """Flood a region using one uint64 bit per cell, without queue allocations."""
    if mask == 0:
        return False
    reached = mask & (~mask + np.uint64(1))
    while True:
        neighbors = ((reached & np.uint64(0xFEFEFEFEFEFEFEFE)) >> np.uint64(1)
                     | (reached & np.uint64(0x7F7F7F7F7F7F7F7F)) << np.uint64(1)
                     | reached >> np.uint64(8) | reached << np.uint64(8))
        expanded = reached | (neighbors & mask)
        if expanded == reached:
            return reached == mask
        reached = expanded


@njit(cache=True)
def placement_histograms(board, candidates):
    """Pack each placement's eight region counts into eight four-bit digits."""
    histograms = np.zeros(len(candidates), dtype=np.uint32)
    for index in range(len(candidates)):
        for row in range(SIZE):
            histograms[index] += np.int64(1) << (4 * board[row, candidates[index, row]])
    return histograms


@njit(cache=True, nogil=True)
def refine_regions(board, columns, candidates, affected, offsets, rng, max_moves=6000):
    """Move boundary cells to eliminate alternatives while keeping region shape constraints."""
    sizes = np.zeros(SIZE, dtype=np.int64)
    masks = np.zeros(SIZE, dtype=np.uint64)
    for row in range(SIZE):
        for col in range(SIZE):
            sizes[board[row, col]] += 1
            masks[board[row, col]] |= np.uint64(1) << np.uint64(row * SIZE + col)
    histograms = placement_histograms(board, candidates)
    current_count = np.count_nonzero(histograms == ALL_REGIONS_ONCE)
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
        bit = np.uint64(1) << np.uint64(cell)
        remaining = masks[old] ^ bit
        if not connected_mask(remaining):
            continue
        change = (np.int64(1) << (4 * new)) - (np.int64(1) << (4 * old))
        proposed_count = current_count
        for position in range(offsets[cell], offsets[cell + 1]):
            index = affected[position]
            previous = np.int64(histograms[index])
            histograms[index] = previous + change
            proposed_count += int(previous + change == ALL_REGIONS_ONCE) - int(previous == ALL_REGIONS_ONCE)
        temperature = 0.1 + 3.0 * (1.0 - (move % 1000) / 1000.0) ** 2
        delta = proposed_count - current_count
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            board[row, col] = new
            masks[old] = remaining
            masks[new] |= bit
            sizes[old] -= 1
            sizes[new] += 1
            current_count = proposed_count
            if current_count == 1:
                return board, move + 1
        else:
            for position in range(offsets[cell], offsets[cell + 1]):
                histograms[affected[position]] -= change
    return board, -1


@njit(cache=True)
def normalize_board(board):
    names = np.full(SIZE, -1, dtype=np.int8)
    output = np.empty((SIZE, SIZE), dtype=np.uint8)
    count = 0
    for row in range(SIZE):
        for col in range(SIZE):
            label = board[row, col]
            if names[label] < 0:
                names[label] = count
                count += 1
            output[row, col] = names[label]
    return output


@njit(cache=True)
def canonical_board(board):
    """Lexicographically smallest normalized rotation/reflection."""
    best = normalize_board(board)
    rotated = board.copy()
    for turns in range(4):
        if turns:
            rotated = np.rot90(rotated).copy()
        for reflected in range(2):
            variant = normalize_board(np.fliplr(rotated) if reflected else rotated)
            for cell in range(SIZE * SIZE):
                value, previous = variant.flat[cell], best.flat[cell]
                if value < previous:
                    best = variant
                    break
                if value > previous:
                    break
    return best


@njit(cache=True, nogil=True)
def generate_batch(attempts, candidates, affected, offsets, rng):
    """Generate independent candidates inside a GIL-free compiled batch."""
    boards = np.empty((attempts, SIZE, SIZE), dtype=np.uint8)
    columns = np.empty((attempts, SIZE), dtype=np.uint8)
    canonical = np.empty_like(boards)
    status = np.zeros(attempts, dtype=np.uint8)
    for index in range(attempts):
        placement = candidates[rng.integers(0, len(candidates))]
        board = grow_regions(placement, rng)
        sizes = np.bincount(board.ravel(), minlength=SIZE)
        if sizes.min() < MIN_REGION_CELLS or sizes.max() > MAX_REGION_CELLS:
            status[index] = 1
            continue
        board, moves = refine_regions(board, placement, candidates, affected, offsets, rng)
        if moves < 0 or not unique_solution(board, candidates):
            status[index] = 2
            continue
        boards[index] = normalize_board(board)
        columns[index] = placement
        canonical[index] = canonical_board(board)
    return boards, columns, canonical, status


@njit(cache=True, nogil=True)
def insert_fingerprints(keys, table, occupied):
    """Exact 256-bit deduplication in a compact open-addressed hash table."""
    fresh = np.empty(len(keys), dtype=np.bool_)
    mask = np.uint64(len(table) - 1)
    for index in range(len(keys)):
        slot = np.int64(keys[index, 0] & mask)
        while occupied[slot]:
            if (table[slot, 0] == keys[index, 0] and table[slot, 1] == keys[index, 1]
                    and table[slot, 2] == keys[index, 2] and table[slot, 3] == keys[index, 3]):
                fresh[index] = False
                break
            slot = (slot + np.int64(1)) & np.int64(mask)
        else:
            table[slot] = keys[index]
            occupied[slot] = True
            fresh[index] = True
    return fresh


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _close_arrays(arrays):
    for array in arrays.values():
        array._mmap.close()


def publish_dataset(staging, output, metadata):
    """Publish one immutable array directory through an atomic manifest update."""
    entries = array_manifest(staging)
    identity = hashlib.sha256()
    for name, entry in entries.items():
        identity.update(name.encode("ascii"))
        identity.update(bytes.fromhex(entry["sha256"]))
    destination = output / f"arrays-{identity.hexdigest()[:16]}-{time.time_ns()}"
    staging.replace(destination)
    for entry in entries.values():
        entry["file"] = f"{destination.name}/{entry['file']}"
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {
        **metadata, "version": DATASET_VERSION, "board_size": SIZE,
        "arrays": entries,
        "data_sha256": metadata.get("data_sha256", identity.hexdigest()),
    }
    _write_json(manifest_path, manifest)
    state_path = destination / "state.json"
    if state_path.exists():
        state_path.unlink()
    for entry in previous.get("arrays", {}).values():
        path = (output / entry["file"]).resolve()
        if path.is_relative_to(output.resolve()) and path.parent != destination.resolve():
            try:
                path.unlink(missing_ok=True)
                if path.parent != output.resolve() and not any(path.parent.iterdir()):
                    path.parent.rmdir()
            except PermissionError:
                print(f"Previous array still open: {path}", flush=True)
    return manifest


def generate(count=100000, seed=1701, max_attempts=None, *, workers=None,
             batch_size=256, output=None, resume=False, checkpoint_seconds=5.0):
    """Generate compact arrays with bounded worker queues and resumable progress."""
    workers = (os.cpu_count() or 1) if workers is None else int(workers)
    if count < 3 or seed < 0 or workers < 1 or batch_size < 1 or checkpoint_seconds <= 0:
        raise ValueError("Use count >= 3, seed >= 0, and positive workers, batch size and save interval.")
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    output = Path(output) if output is not None else PROJECT_ROOT / "data" / "queens8"
    output.mkdir(parents=True, exist_ok=True)
    staging = output / ".generation"
    state_path = staging / "state.json"
    config = dict(count=count, seed=seed, batch_size=batch_size, algorithm="incremental_v1")
    arrays = {}
    try:
        if resume:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["config"] != config:
                raise ValueError("Resume requires the same count, seed, batch size and algorithm.")
            for name, shape in ARRAY_SHAPES.items():
                arrays[name] = np.load(staging / f"{name}.npy", mmap_mode="r+", allow_pickle=False)
                if arrays[name].shape != (count, *shape) or arrays[name].dtype != np.uint8:
                    raise ValueError(f"Invalid staged array: {name}.")
        else:
            if staging.exists():
                raise FileExistsError(f"Unfinished generation at {staging}; use --resume or another --output.")
            staging.mkdir()
            for name, shape in ARRAY_SHAPES.items():
                arrays[name] = np.lib.format.open_memmap(
                    staging / f"{name}.npy", mode="w+", dtype=np.uint8, shape=(count, *shape),
                )
            state = dict(config=config, written=0, attempts=0, next_batch=0,
                         batch_offset=0, rejected={}, elapsed=0.0)
            _write_json(state_path, state)

        candidates, affected, offsets = prepare_candidates()
        print("Compiling CPU kernels...", flush=True)
        generate_batch(0, candidates, affected, offsets, np.random.default_rng(0))
        capacity = 1 << max(3, math.ceil(math.log2((count + batch_size) / 0.7)))
        table = np.empty((capacity, 4), dtype=np.uint64)
        occupied = np.zeros(capacity, dtype=np.bool_)
        insert_fingerprints(np.frombuffer(b"", np.uint64).reshape(0, 4), table, occupied)
        for start in range(0, state["written"], 65536):
            stored = arrays["fingerprints"][start:min(start + 65536, state["written"])]
            if not np.all(insert_fingerprints(stored.view(np.uint64).reshape(-1, 4), table, occupied)):
                raise ValueError("Duplicate fingerprints in the partial dataset.")

        limit = max_attempts if max_attempts is not None else count * 1000
        rejected = Counter(state["rejected"])
        started = last_save = time.perf_counter()
        previous_elapsed = state["elapsed"]
        initial_written = state["written"]
        pending = deque()
        scheduled = state["attempts"]
        queued_attempts = 0
        next_batch = state["next_batch"]
        next_offset = state["batch_offset"]
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="queens")

        def checkpoint():
            for array in arrays.values():
                array.flush()
            state["rejected"] = dict(rejected)
            state["elapsed"] = previous_elapsed + time.perf_counter() - started
            _write_json(state_path, state)

        def submit():
            nonlocal scheduled, next_batch, next_offset, queued_attempts
            size = min(batch_size - next_offset, limit - scheduled)
            rng = np.random.default_rng(np.random.SeedSequence([seed, next_batch]))
            pending.append((next_batch, next_offset, size, executor.submit(
                generate_batch, batch_size, candidates, affected, offsets, rng,
            )))
            scheduled += size
            queued_attempts += size
            next_batch += 1
            next_offset = 0

        print(f"Preparing {count:,} puzzles | {workers} CPU workers | batch {batch_size}", flush=True)
        try:
            while state["written"] < count and (pending or scheduled < limit):
                while (len(pending) < 2 * workers and scheduled < limit
                       and queued_attempts < max(batch_size, count - state["written"]) * 1.25):
                    submit()
                batch_index, batch_offset, size, future = pending.popleft()
                queued_attempts -= size
                boards, columns, canonical, status = future.result()
                part = slice(batch_offset, batch_offset + size)
                boards, columns = boards[part], columns[part]
                canonical, status = canonical[part], status[part]
                valid = np.flatnonzero(status == 0)
                digests = b"".join(hashlib.sha256(canonical[index].tobytes()).digest() for index in valid)
                fingerprints = np.frombuffer(digests, dtype=np.uint8).reshape(-1, 32)
                fresh = insert_fingerprints(fingerprints.view(np.uint64).reshape(-1, 4), table, occupied)
                selected = valid[fresh][:count - state["written"]]
                end = state["written"] + len(selected)
                arrays["regions"][state["written"]:end] = boards[selected]
                arrays["queen_columns"][state["written"]:end] = columns[selected]
                arrays["fingerprints"][state["written"]:end] = fingerprints[fresh][:len(selected)]
                rejected["region_size"] += int(np.count_nonzero(status == 1))
                rejected["not_unique"] += int(np.count_nonzero(status == 2))
                rejected["symmetry_duplicate"] += int(np.count_nonzero(~fresh))
                state.update(written=end, attempts=state["attempts"] + size,
                             next_batch=batch_index + (batch_offset + size) // batch_size,
                             batch_offset=(batch_offset + size) % batch_size)
                now = time.perf_counter()
                if now - last_save >= checkpoint_seconds or end == count:
                    checkpoint()
                    speed = (end - initial_written) / max(now - started, 1e-9)
                    remaining = (count - end) / max(speed, 1e-9)
                    print(f"{end:,}/{count:,} | {speed:,.0f} puzzles/s | "
                          f"ETA {remaining / 60:.1f} min | {state['attempts']:,} candidates", flush=True)
                    last_save = now
            checkpoint()
        finally:
            for _, _, _, future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        if state["written"] != count:
            raise RuntimeError(f"Generated {state['written']:,}/{count:,}; partial arrays saved. "
                               "Increase --max-attempts and use --resume.")
    finally:
        _close_arrays(arrays)

    manifest = publish_dataset(staging, output, dict(
        count=count, seed=seed, prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        generator="scripts/generate_queens8.py", candidate_boards=state["attempts"],
        region_cells=dict(min=MIN_REGION_CELLS, max=MAX_REGION_CELLS),
        rules="One queen per row, column and region; queens cannot touch diagonally.",
        growth="One adjacent cell per region per shuffled round; then connected boundary refinement.",
        uniqueness="Exactly one solution: all non-touching row/column permutations checked offline.",
        symmetry_deduplicated=True, rejected_candidates=dict(rejected),
        workers=workers, batch_size=batch_size, generation_seconds=state["elapsed"],
    ))
    print(f"Saved {output} | SHA256 {manifest['data_sha256']}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "queens8")
    parser.add_argument("--resume", action="store_true", help="Continue saved partial array generation.")
    args = parser.parse_args()
    if args.count < 3:
        parser.error("--count must be at least 3")
    generate(args.count, args.seed, args.max_attempts, workers=args.workers,
             batch_size=args.batch_size, output=args.output, resume=args.resume)
