"""Load prepared synthetic Queens pairs and check proposed solutions."""

import hashlib
import json
from pathlib import Path

import numpy as np


BOARD_SIZE = 8
EMPTY = BOARD_SIZE
ARRAY_SHAPES = {
    "regions": (BOARD_SIZE, BOARD_SIZE),
    "queen_columns": (BOARD_SIZE,),
    "fingerprints": (32,),
}
DATASET_VERSION = "queens8_arrays_v1"
DEFAULT_VALIDATION_PUZZLES = 1024
DEFAULT_TEST_PUZZLES = 256


def file_sha256(path):
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_manifest(directory):
    """Describe the three compact arrays and their storage checksums."""
    directory = Path(directory)
    return {
        name: dict(file=f"{name}.npy", sha256=file_sha256(directory / f"{name}.npy"))
        for name in ARRAY_SHAPES
    }


class PuzzleIds:
    """Materialize readable identifiers only for the requested puzzle indices."""

    def __init__(self, fingerprints):
        self.fingerprints = fingerprints

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, index):
        selected = self.fingerprints[index]
        if selected.ndim == 1:
            return f"q8_{selected[:8].tobytes().hex()}"
        ids = [f"q8_{digest[:8].tobytes().hex()}" for digest in selected.reshape(-1, 32)]
        return np.asarray(ids, dtype="U19").reshape(selected.shape[:-1])


def validate_pairs(regions, queen_columns, chunk_size=32768):
    """Check stored answers in bounded batches without constructing grid masks."""
    expected = np.arange(BOARD_SIZE)
    for start in range(0, len(regions), chunk_size):
        boards = regions[start:start + chunk_size]
        columns = queen_columns[start:start + chunk_size]
        if np.any(boards >= BOARD_SIZE) or np.any(columns >= BOARD_SIZE):
            raise ValueError(f"Invalid region or queen coordinate near puzzle {start}.")
        colors = boards[np.arange(len(boards))[:, None], expected, columns]
        if (not np.all(np.sort(columns, axis=1) == expected)
                or not np.all(np.sort(colors, axis=1) == expected)
                or np.any(np.abs(np.diff(columns.astype(np.int16), axis=1)) == 1)):
            raise ValueError(f"Stored answer breaks the Queens rules near puzzle {start}.")


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


def load_queens_dataset(directory, *, verify=True):
    """Memory-map compact pairs; optionally stream checksums and validate answers."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (metadata.get("board_size") != BOARD_SIZE or metadata.get("version") != DATASET_VERSION
            or metadata.get("count", 0) < 3):
        raise ValueError("Expected a compact synthetic 8x8 Queens dataset.")
    arrays = {}
    try:
        for name, shape in ARRAY_SHAPES.items():
            entry = metadata["arrays"][name]
            path = directory / entry["file"]
            array = arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.uint8 or array.shape != (metadata["count"], *shape):
                raise ValueError(f"Invalid shape or dtype for {name}.")
            if verify and file_sha256(path) != entry["sha256"]:
                raise ValueError(f"{name} does not match its manifest checksum.")
        if verify:
            validate_pairs(arrays["regions"], arrays["queen_columns"])
    except BaseException:
        for array in arrays.values():
            array._mmap.close()
        raise
    return {**arrays, "ids": PuzzleIds(arrays["fingerprints"]), "metadata": metadata}


def split_puzzles(count, seed=17, validation_fraction=0.15, test_fraction=0.2):
    """Split generator-deduplicated puzzle indices before augmentation."""
    count = int(count)
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Fractions must leave nonempty train, validation, and test sets.")
    order = np.random.default_rng(seed).permutation(count)
    n_validation = max(1, round(len(order) * validation_fraction))
    n_test = max(1, round(len(order) * test_fraction))
    if n_validation + n_test >= len(order):
        raise ValueError("Not enough puzzles for the requested split.")
    return {
        "test": order[:n_test],
        "validation": order[n_test : n_test + n_validation],
        "train": order[n_test + n_validation :],
    }


def fixed_evaluation_splits(splits, *, validation_puzzles=DEFAULT_VALIDATION_PUZZLES,
                            test_puzzles=DEFAULT_TEST_PUZZLES):
    """Select fixed prefixes of the seeded, randomly ordered held-out splits."""
    if validation_puzzles < 1 or test_puzzles < 1:
        raise ValueError("Validation and test subset sizes must be positive.")
    return {
        "train": splits["train"],
        "validation": splits["validation"][:int(validation_puzzles)],
        "test": splits["test"][:int(test_puzzles)],
    }


def solution_checks(regions, tokens, empty=EMPTY):
    """Check the Queens rules for supplied tokens in O(N^2).

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

