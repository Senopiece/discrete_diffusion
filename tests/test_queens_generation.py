"""Exact generator checks, parallel reproducibility, and compact data integration."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "notebooks"))

import generate_queens8 as generation
from _shared.queens_data import (
    ARRAY_SHAPES, load_queens_dataset, normalize_regions, puzzle_fingerprint,
    split_puzzles, validate_pairs,
)


def close_dataset(puzzles):
    for name in ARRAY_SHAPES:
        puzzles[name]._mmap.close()


class ExactGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates, cls.affected, cls.offsets = generation.prepare_candidates()

    def test_incremental_counts_equal_exhaustive_counts(self):
        rng = np.random.default_rng(1781)
        board = generation.grow_regions(self.candidates[173], rng)
        histograms = generation.placement_histograms(board, self.candidates)
        for _ in range(250):
            cell = int(rng.integers(64))
            row, col = divmod(cell, 8)
            old, new = int(board[row, col]), int(rng.integers(8))
            indices = self.affected[self.offsets[cell]:self.offsets[cell + 1]]
            histograms[indices] = (
                histograms[indices].astype(np.int64) + (1 << (4 * new)) - (1 << (4 * old))
            ).astype(np.uint32)
            board[row, col] = new
            recomputed = generation.placement_histograms(board, self.candidates)
            np.testing.assert_array_equal(histograms, recomputed)
            self.assertEqual(np.count_nonzero(histograms == generation.ALL_REGIONS_ONCE),
                             generation.solution_count(board, self.candidates))

    def test_bit_flood_matches_graph_search(self):
        rng = np.random.default_rng(139)
        for _ in range(250):
            cells = set(np.flatnonzero(rng.random(64) < rng.uniform(0.02, 0.8)).tolist())
            reached = set()
            pending = [next(iter(cells))] if cells else []
            while pending:
                cell = pending.pop()
                if cell in reached:
                    continue
                reached.add(cell)
                row, col = divmod(cell, 8)
                for dr, dc in generation.DIRECTIONS:
                    nr, nc = row + dr, col + dc
                    neighbor = nr * 8 + nc
                    if 0 <= nr < 8 and 0 <= nc < 8 and neighbor in cells - reached:
                        pending.append(neighbor)
            mask = np.uint64(sum(1 << cell for cell in cells))
            self.assertEqual(generation.connected_mask(mask), bool(cells) and reached == cells)

    def test_generated_pairs_have_unique_solutions_and_valid_regions(self):
        boards, columns, canonical, status = generation.generate_batch(
            80, self.candidates, self.affected, self.offsets, np.random.default_rng(91),
        )
        valid = status == 0
        self.assertGreater(np.count_nonzero(valid), 30)
        validate_pairs(boards[valid], columns[valid])
        for index in np.flatnonzero(valid):
            board = boards[index]
            self.assertEqual(generation.solution_count(board, self.candidates), 1)
            np.testing.assert_array_equal(board, normalize_regions(board))
            self.assertEqual(hashlib.sha256(canonical[index].tobytes()).hexdigest(),
                             puzzle_fingerprint(board))
            sizes = np.bincount(board.ravel(), minlength=8)
            self.assertGreaterEqual(sizes.min(), generation.MIN_REGION_CELLS)
            self.assertLessEqual(sizes.max(), generation.MAX_REGION_CELLS)
            for label in range(8):
                mask = np.uint64(sum(1 << int(cell) for cell in np.flatnonzero(board.ravel() == label)))
                self.assertTrue(generation.connected_mask(mask))

    def test_deduplication_handles_collisions_and_zero_digest(self):
        keys = np.array([[0, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0],
                         [0, 0, 0, 1], [8, 0, 0, 0]], dtype=np.uint64)
        fresh = generation.insert_fingerprints(keys, np.empty((8, 4), np.uint64), np.zeros(8, bool))
        np.testing.assert_array_equal(fresh, [True, True, False, True, False])


class DatasetIntegrationTests(unittest.TestCase):
    def test_parallel_reproducibility_and_lazy_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "one", Path(temporary) / "four"
            a = generation.generate(80, 921, workers=1, batch_size=16, output=left)
            b = generation.generate(80, 921, workers=4, batch_size=16, output=right)
            self.assertEqual(a["data_sha256"], b["data_sha256"])
            puzzles = load_queens_dataset(left)
            try:
                for name in ARRAY_SHAPES:
                    self.assertIsInstance(puzzles[name], np.memmap)
                    self.assertFalse(puzzles[name].flags.writeable)
                indices = np.array([0, 7, 15])
                ids = puzzles["ids"][indices]
                self.assertEqual(ids[0], puzzles["ids"][0])
                for index, puzzle_id in zip(indices, ids):
                    self.assertEqual(puzzle_id, "q8_" + puzzles["fingerprints"][index, :8].tobytes().hex())
                split = split_puzzles(80, seed=17)
                self.assertEqual(len(set(np.concatenate(list(split.values())))), 80)
                from _shared.queens_dfm_training import _QueensViews
                board, endpoint = _QueensViews(puzzles, indices, False)[0]
                self.assertEqual(tuple(board.shape), (8, 8))
                self.assertEqual(int((endpoint != 8).sum()), 8)
            finally:
                close_dataset(puzzles)

    def test_resume_and_completed_dataset_survives_partial_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            existing = generation.generate(16, 77, workers=1, batch_size=8, output=output)
            with self.assertRaises(RuntimeError):
                generation.generate(48, 78, max_attempts=13, workers=2, batch_size=8, output=output)
            metadata = json.loads((output / "manifest.json").read_text())
            self.assertEqual(metadata["data_sha256"], existing["data_sha256"])
            completed = generation.generate(48, 78, workers=4, batch_size=8, output=output, resume=True)
            fresh = generation.generate(48, 78, workers=1, batch_size=8, output=output / "fresh")
            self.assertEqual(completed["data_sha256"], fresh["data_sha256"])
            puzzles = load_queens_dataset(output)
            try:
                self.assertEqual(len(puzzles["regions"]), 48)
                self.assertNotEqual(completed["data_sha256"], existing["data_sha256"])
                self.assertFalse((output / ".generation").exists())
            finally:
                close_dataset(puzzles)

    def test_storage_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            metadata = generation.generate(3, 17, workers=2, output=output)
            path = output / metadata["arrays"]["regions"]["file"]
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(bytes([99]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_queens_dataset(output)


if __name__ == "__main__":
    unittest.main()
