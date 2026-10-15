"""Load curated, already solved Queens puzzles and score proposed solutions.

There is deliberately no puzzle generator, solver, or uniqueness search here.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


BOARD_SIZE = 7
EMPTY = BOARD_SIZE


def normalize_regions(regions):
    """Rename regions by first occurrence, independently of the source colors."""
    array = np.asarray(regions)
    names = {}
    result = np.empty(array.shape, dtype=np.uint8)
    for index in np.ndindex(array.shape):
        label = array[index].item()
        if label not in names:
            names[label] = len(names)
        result[index] = names[label]
    return result


def puzzle_fingerprint(regions):
    """Identify duplicates up to rotations, reflections, and color relabeling."""
    variants = []
    for turns in range(4):
        rotated = np.rot90(regions, turns)
        variants.append(normalize_regions(rotated).tobytes())
        variants.append(normalize_regions(np.fliplr(rotated)).tobytes())
    return hashlib.sha256(min(variants)).hexdigest()


def load_queens_dataset(directory):
    """Read the local multi-source JSONL pairs; use their published solutions."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    raw_data = (directory / "puzzles.jsonl").read_bytes()
    if hashlib.sha256(raw_data).hexdigest() != metadata["data_sha256"]:
        raise ValueError("Dataset does not match its manifest; rebuild both files together.")
    records = [json.loads(line) for line in raw_data.decode("utf-8").splitlines() if line.strip()]
    boards, targets = [], []
    for record in records:
        board = np.asarray(record["regions"], dtype=np.uint8)
        coordinates = np.asarray(record["queens"], dtype=np.int64)
        if board.shape != (BOARD_SIZE, BOARD_SIZE) or coordinates.shape != (BOARD_SIZE, 2):
            raise ValueError(f"Malformed stored pair: {record['id']}")
        queens = np.zeros_like(board, dtype=bool)
        queens[coordinates[:, 0], coordinates[:, 1]] = True
        boards.append(board)
        targets.append(queens)
    return {
        "regions": np.stack(boards),
        "queens": np.stack(targets),
        "source_ids": np.asarray([record["id"] for record in records]),
        "source_names": np.asarray([record["sources"][0]["name"] for record in records]),
        "fingerprints": np.asarray([record["fingerprint"] for record in records]),
        "records": records,
        "metadata": metadata,
    }


def split_puzzles(fingerprints, seed=17, validation_fraction=0.15, test_fraction=0.2):
    """Split distinct puzzles before augmentation, without symmetry leakage."""
    fingerprints = np.asarray(fingerprints)
    if len(set(fingerprints.tolist())) != len(fingerprints):
        raise ValueError("Deduplicate puzzle symmetries before splitting.")
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Fractions must leave nonempty train, validation, and test sets.")
    order = np.random.default_rng(seed).permutation(len(fingerprints))
    n_validation = max(1, round(len(order) * validation_fraction))
    n_test = max(1, round(len(order) * test_fraction))
    if n_validation + n_test >= len(order):
        raise ValueError("Not enough puzzles for the requested split.")
    return {
        "test": order[:n_test],
        "validation": order[n_test : n_test + n_validation],
        "train": order[n_test + n_validation :],
    }


def solution_checks(regions, tokens, empty=EMPTY):
    """Check a supplied output in O(N^2); no search, uniqueness check, or repair.

    Matching batch dimensions can precede (N,N). Every nonempty cell is a
    predicted queen and must retain its original region color.
    """
    regions, tokens = np.asarray(regions), np.asarray(tokens)
    if regions.shape != tokens.shape or regions.shape[-2] != regions.shape[-1]:
        raise ValueError("Expected matching arrays ending in square board dimensions.")
    size = regions.shape[-1]
    occupied = tokens != empty
    colors = np.all((tokens == empty) | (tokens == regions), axis=(-2, -1))
    rows = np.all(occupied.sum(axis=-1) == 1, axis=-1)
    columns = np.all(occupied.sum(axis=-2) == 1, axis=-1)
    region_counts = np.zeros((*regions.shape[:-2], size), dtype=np.int64)
    flat_regions = regions.reshape(-1, size * size)
    flat_occupied = occupied.reshape(-1, size * size)
    flat_counts = region_counts.reshape(-1, size)
    for index in range(len(flat_regions)):
        flat_counts[index] = np.bincount(
            flat_regions[index], weights=flat_occupied[index], minlength=size
        )
    distinct_regions = np.all(region_counts == 1, axis=-1)
    touching = (
        np.any(occupied[..., :-1, :-1] & occupied[..., 1:, 1:], axis=(-2, -1))
        | np.any(occupied[..., :-1, 1:] & occupied[..., 1:, :-1], axis=(-2, -1))
    )
    return {
        "solved": colors & rows & columns & distinct_regions & ~touching,
        "correct_colors": colors,
        "one_per_row": rows,
        "one_per_column": columns,
        "one_per_region": distinct_regions,
        "no_touching": ~touching,
        "queen_count": occupied.sum(axis=(-2, -1)),
    }

