"""Fixed evaluation subsets and sample-weighted epoch metrics."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from _shared.queens_data import fixed_evaluation_splits, split_puzzles
from _shared.queens_dfm_training import QueensDFMRun, epoch_loss_history
from _shared.queens_confidence_sampling import load_trained_dfm


class TinyDFM(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))
        self.use_time_conditioning = config.get("use_time_conditioning", True)
        self.training_batches = []
        self.validation_samples = 0

    def forward(self, state, times=None):
        if self.training:
            self.training_batches.append(len(state))
        else:
            self.validation_samples += len(state)
        return self.logit.expand(len(state), 1, 8, 8)


def pairs(count=200):
    return dict(
        regions=np.tile(np.arange(8, dtype=np.uint8)[None, :, None], (count, 1, 8)),
        queen_columns=np.tile(np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.uint8), (count, 1)),
        metadata=dict(data_sha256="training-test"),
    )


class TrainingTests(unittest.TestCase):
    def test_fixed_subsets_are_disjoint_and_repeatable(self):
        full = split_puzzles(200, seed=17)
        selected = fixed_evaluation_splits(full, validation_puzzles=7, test_puzzles=9)
        repeated = fixed_evaluation_splits(split_puzzles(200, seed=17), validation_puzzles=7, test_puzzles=9)
        self.assertEqual(len(selected["validation"]), 7)
        self.assertEqual(len(selected["test"]), 9)
        for name in selected:
            np.testing.assert_array_equal(selected[name], repeated[name])
        all_indices = np.concatenate(list(selected.values()))
        self.assertEqual(len(np.unique(all_indices)), len(all_indices))
        small = fixed_evaluation_splits(split_puzzles(10))
        self.assertEqual(len(small["test"]), 2)

    def test_epoch_means_weight_short_batch_and_omit_partial_epoch(self):
        history = dict(train_bce=[1.0, 3.0, 9.0, 4.0], validation_step=[3], validation_bce=[0.7])
        result = epoch_loss_history(history, samples_per_epoch=10, batch_size=4)
        np.testing.assert_array_equal(result["epoch"], [1])
        np.testing.assert_allclose(result["train_bce"], [3.4])
        np.testing.assert_array_equal(result["validation_epoch"], [1])
        empty = epoch_loss_history(dict(train_bce=[], validation_step=[], validation_bce=[]),
                                   samples_per_epoch=10, batch_size=4)
        self.assertEqual(len(empty["epoch"]), 0)

    def test_resume_preserves_partial_epoch_and_inference_uses_saved_subsets(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("_shared.queens_dfm_training.QueensDFM", TinyDFM), \
                patch("_shared.queens_confidence_sampling.QueensDFM", TinyDFM):
            path = Path(directory) / "model.pt"
            puzzles = pairs()
            config = dict(batch_size=4, samples_per_epoch=10, epochs=2,
                          validation_puzzles=7, test_puzzles=9, use_time_conditioning=False)
            first = QueensDFMRun(puzzles, path, **config)
            first.train(target_steps=1)
            self.assertEqual(first.history["epochs_completed"], 0)
            resumed = QueensDFMRun(puzzles, path, **config)
            self.assertTrue(resumed.resumed)
            self.assertEqual(resumed.global_step, 1)
            resumed.train()
            self.assertEqual(first.model.training_batches + resumed.model.training_batches, [4, 4, 2, 4, 4, 2])
            self.assertEqual(resumed.model.validation_samples, 2 * 7 * 6)
            self.assertEqual(resumed.history["epochs_completed"], 2)
            self.assertEqual(len(epoch_loss_history(resumed.history, samples_per_epoch=10, batch_size=4)["epoch"]), 2)
            _, selected, info = load_trained_dfm(path, puzzles)
            self.assertEqual(info["test_puzzles"], 9)
            for name in selected:
                np.testing.assert_array_equal(selected[name], resumed.evaluation_splits[name])

            saved = torch.load(path, weights_only=True)
            saved.pop("evaluation_config")
            torch.save(saved, path)
            loaded = QueensDFMRun(puzzles, path, **config)
            self.assertEqual(loaded.global_step, resumed.global_step)
            self.assertEqual(loaded.history["train_bce"], resumed.history["train_bce"])
            self.assertEqual(loaded.history["validation_bce"], [])
            torch.testing.assert_close(loaded.model.logit, resumed.model.logit)


if __name__ == "__main__":
    unittest.main()
