"""A second, independent Queens DFM training run with isolated random state."""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .queens_data import BOARD_SIZE, EMPTY
from .queens_dfm_inference import QueensDFM


class _QueensViews(Dataset):
    def __init__(self, puzzles, indices, augment):
        self.puzzles, self.indices, self.augment = puzzles, np.asarray(indices), augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        board = torch.from_numpy(self.puzzles["regions"][index].astype(np.int64))
        queens = torch.from_numpy(self.puzzles["queens"][index].copy())
        if self.augment:
            turns = int(torch.randint(4, ()).item())
            board, queens = torch.rot90(board, turns), torch.rot90(queens, turns)
            if bool(torch.randint(2, ()).item()):
                board, queens = board.flip(-1), queens.flip(-1)
            board = torch.randperm(BOARD_SIZE)[board]
        return board.contiguous(), torch.where(queens, board, EMPTY).contiguous()


def _path(source, target, times, generator=None):
    switches = torch.rand(source.shape, device=source.device, generator=generator)
    return torch.where(switches < times[:, None, None], target, source)


class IndependentQueensDFM:
    """Same CE objective/augmentation as DFM #1; own weights, optimizer and RNG."""

    def __init__(self, puzzles, reference_path, checkpoint_path, *, device="cpu",
                 seed=None, batch_size=128, samples_per_epoch=4096, resume=True):
        self.puzzles = puzzles
        self.reference_path, self.checkpoint_path = Path(reference_path), Path(checkpoint_path)
        if self.reference_path.resolve() == self.checkpoint_path.resolve():
            raise ValueError("The second run needs a separate checkpoint path.")
        reference = torch.load(self.reference_path, map_location="cpu", weights_only=True)
        if reference.get("format_version") != "queens7_dfm_v1":
            raise ValueError("Expected a trained Queens DFM reference checkpoint.")
        if reference.get("dataset_sha256") != puzzles["metadata"]["data_sha256"]:
            raise ValueError("Reference checkpoint and dataset differ.")
        self.target_steps = int(reference["global_step"])
        if self.target_steps < 1:
            raise ValueError("Train and save the first model before starting the second.")
        self.reference_epochs = int(reference["history"]["epochs_completed"])
        self.reference_history = reference["history"]
        self.reference_seed = int(reference["seed"])
        self.seed = self.reference_seed + 1 if seed is None else int(seed)
        if self.seed == self.reference_seed:
            raise ValueError("Choose a different seed for the independent run.")
        self.device = torch.device(device)
        self.cuda_devices = list(range(torch.cuda.device_count())) if self.device.type == "cuda" else []
        self.batch_size, self.samples_per_epoch = int(batch_size), int(samples_per_epoch)
        if self.batch_size < 1 or self.samples_per_epoch < 1:
            raise ValueError("Batch size and samples per epoch must be positive.")
        self.steps_per_epoch = math.ceil(self.samples_per_epoch / self.batch_size)
        self.model_config = dict(reference["model_config"])
        self.flow_steps = int(reference["flow_steps"])
        self.splits = {name: indices.cpu().numpy().copy() for name, indices in reference["splits"].items()}
        parameters = reference["optimizer_state_dict"]["param_groups"][0]
        self.learning_rate = float(parameters["lr"])
        self.weight_decay = float(parameters["weight_decay"])
        with self.reference_path.open("rb") as stream:
            self.reference_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.history = dict(step=[], train_ce=[], validation_step=[], validation_ce=[],
                            validation_t0_ce=[], epochs_completed=0)
        self.global_step = 0
        self.resumed = False
        with torch.random.fork_rng(devices=self.cuda_devices):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(self.seed).get_state())
            if self.cuda_devices:
                torch.cuda.manual_seed_all(self.seed)
            self.model = QueensDFM(**self.model_config).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
            )
            self._capture_rng()
        self.model.eval()
        if resume and self.checkpoint_path.exists():
            self.load()

    def _capture_rng(self):
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if self.cuda_devices else []

    def _training_config(self):
        return dict(batch_size=self.batch_size, samples_per_epoch=self.samples_per_epoch,
                    learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                    validation_seed=self.reference_seed + 401)

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(".pt.tmp")
        torch.save(dict(
            format_version="queens7_dfm_v1", model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(), model_config=self.model_config,
            history=self.history, global_step=self.global_step, flow_steps=self.flow_steps,
            seed=self.seed, split_seed=self.reference_seed,
            dataset_sha256=self.puzzles["metadata"]["data_sha256"],
            splits={name: torch.as_tensor(indices.copy()) for name, indices in self.splits.items()},
            torch_rng_state=self.cpu_rng, cuda_rng_state_all=self.cuda_rng,
            training_config=self._training_config(), independent_run=True,
            reference_checkpoint_sha256=self.reference_sha256,
            target_optimizer_steps=self.target_steps,
        ), temporary)
        temporary.replace(self.checkpoint_path)

    def load(self):
        saved = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if (
            saved.get("format_version") != "queens7_dfm_v1"
            or not saved.get("independent_run")
            or saved.get("dataset_sha256") != self.puzzles["metadata"]["data_sha256"]
            or int(saved["seed"]) != self.seed
            or saved["model_config"] != self.model_config
            or saved.get("training_config") != self._training_config()
            or any(not np.array_equal(saved["splits"][name].cpu().numpy(), values)
                   for name, values in self.splits.items())
        ):
            raise ValueError("Second checkpoint/settings differ; choose the matching settings or a new path.")
        self.model.load_state_dict(saved["model_state_dict"])
        self.optimizer.load_state_dict(saved["optimizer_state_dict"])
        self.history, self.global_step = saved["history"], int(saved["global_step"])
        self.cpu_rng = saved["torch_rng_state"].clone()
        self.cuda_rng = saved["cuda_rng_state_all"] if self.cuda_devices else []
        if self.cuda_devices and len(self.cuda_rng) != len(self.cuda_devices):
            raise ValueError("CUDA RNG state/device count differs from the saved second run.")
        self.model.eval()
        self.resumed = True

    @torch.no_grad()
    def _validate(self, loader):
        was_training = self.model.training
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(self.reference_seed + 401)
        total, count, at_zero, count_zero = 0.0, 0, 0.0, 0
        try:
            for coloring, target in loader:
                coloring, target = coloring.to(self.device), target.to(self.device)
                for time in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
                    times = torch.full((len(coloring),), time, device=self.device)
                    state = _path(coloring, target, times, generator)
                    loss = F.cross_entropy(self.model(state, coloring, times), target).item()
                    total += loss * len(coloring)
                    count += len(coloring)
                    if time == 0.0:
                        at_zero += loss * len(coloring)
                        count_zero += len(coloring)
            return total / count, at_zero / count_zero
        finally:
            self.model.train(was_training)

    def train(self, *, target_steps=None, checkpoint_every_epochs=20, progress=None):
        """Continue up to a total update budget; auto-save at intervals/end/interruption."""
        target = self.target_steps if target_steps is None else int(target_steps)
        if target < 1 or checkpoint_every_epochs < 0:
            raise ValueError("Use positive target steps and nonnegative save interval.")
        training = _QueensViews(self.puzzles, self.splits["train"], True)
        validation = _QueensViews(self.puzzles, self.splits["validation"], False)
        train_loader = DataLoader(
            training, batch_size=self.batch_size,
            sampler=RandomSampler(training, replacement=True, num_samples=self.samples_per_epoch),
            num_workers=0, pin_memory=self.device.type == "cuda",
        )
        validation_loader = DataLoader(validation, batch_size=self.batch_size, shuffle=False, num_workers=0)
        if self.global_step >= target:
            if progress:
                progress(self, "complete", target)
            return
        with torch.random.fork_rng(devices=self.cuda_devices):
            torch.set_rng_state(self.cpu_rng)
            if self.cuda_devices:
                torch.cuda.set_rng_state_all(self.cuda_rng)
            try:
                while self.global_step < target:
                    self.model.train()
                    for coloring, endpoint in train_loader:
                        if self.global_step >= target:
                            break
                        coloring, endpoint = coloring.to(self.device), endpoint.to(self.device)
                        times = torch.rand(len(coloring), device=self.device)
                        state = _path(coloring, endpoint, times)
                        self.optimizer.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(self.model(state, coloring, times), endpoint)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.global_step += 1
                        self.history["step"].append(self.global_step)
                        self.history["train_ce"].append(float(loss.item()))
                        if progress:
                            progress(self, "step", target)
                        if self.global_step % self.steps_per_epoch == 0:
                            val_ce, val_t0 = self._validate(validation_loader)
                            self.history["validation_step"].append(self.global_step)
                            self.history["validation_ce"].append(val_ce)
                            self.history["validation_t0_ce"].append(val_t0)
                            self.history["epochs_completed"] = self.global_step // self.steps_per_epoch
                            if progress:
                                progress(self, "validation", target)
                            if checkpoint_every_epochs and self.history["epochs_completed"] % checkpoint_every_epochs == 0:
                                self._capture_rng()
                                self.save()
            finally:
                self._capture_rng()
                self.model.eval()
                self.save()
        if progress:
            progress(self, "complete", target)
