"""Resumable NCM training with explicit optimizer-update budgets."""

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from .ncm import GridNCM, PAPER, REFERENCE


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_baseline(factory, batch, loss_fn, *, device, updates=8, warmup=2, lr=2e-4):
    """Time real forward/backward/AdamW updates of a disposable baseline.

    No existing baseline weights/checkpoints are modified. This is a speed
    measurement, not an extra fitted comparison model. FP32 matches old loops.
    """
    device = torch.device(device)
    devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    durations = []
    with torch.random.fork_rng(devices=devices):
        candidate = factory().to(device).train()
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=lr, weight_decay=1e-4)
        parameters = sum(p.numel() for p in candidate.parameters())
        for step in range(warmup + updates):
            synchronize(device)
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(candidate, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
            optimizer.step()
            synchronize(device)
            if step >= warmup:
                durations.append(time.perf_counter() - begin)
        del candidate, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(parameters=parameters, median_update_seconds=float(np.median(durations)),
                measured_update_seconds=durations, precision="float32",
                device=str(device), gpu=(torch.cuda.get_device_name(device) if device.type == "cuda" else None))


class NCMRun:
    FORMAT = "grid_ncm_v1"

    def __init__(self, config, checkpoint_path, *, data_signature, seed, device,
                 learning_rate=2e-4, batch_size=128, micro_batch_size=8):
        self.config = config
        self.path = Path(checkpoint_path)
        self.signature = copy.deepcopy(data_signature)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.micro_batch_size = int(micro_batch_size)
        if min(self.batch_size, self.micro_batch_size) < 1:
            raise ValueError("Batch sizes must be positive.")
        self.learning_rate = learning_rate
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.model = GridNCM(config).to(self.device)
        self.optimizers = {
            "tokenizer": torch.optim.AdamW(
                [p for p in self.model.tokenizer.parameters() if p.requires_grad],
                lr=learning_rate, betas=(.9, .99), weight_decay=.01),
            "generator": torch.optim.AdamW(self.model.generator.parameters(),
                lr=learning_rate, betas=(.9, .99), weight_decay=.01),
        }
        self.plan = None
        self.history = {stage: [] for stage in self.optimizers}
        self.state = {stage: {"updates": 0, "examples": 0, "seconds": 0.}
                      for stage in self.optimizers}
        self.rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
                    if self.device.type == "cuda" else []}
        # The training stream is independent of earlier notebook previews.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 107)
            self.rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
                        if self.device.type == "cuda" else []}

    def set_budget(self, baseline, baseline_updates, tokenizer_fraction=.25):
        if not 0 < tokenizer_fraction < 1 or baseline_updates < 1:
            raise ValueError("Invalid reference budget.")
        seconds = float(baseline["median_update_seconds"] * baseline_updates)
        self.plan = dict(baseline=baseline, baseline_updates=int(baseline_updates),
                         total_seconds=seconds, tokenizer_seconds=seconds * tokenizer_fraction,
                         generator_seconds=seconds * (1 - tokenizer_fraction),
                         tokenizer_fraction=float(tokenizer_fraction))

    def configure_runtime(self):
        precision = (self.plan or {}).get("generator_precision", "float32")
        if precision == "bfloat16" and (self.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            precision = "float32"
        self.model.generator_precision = precision
        self.model.pack_levels = (self.plan or {}).get("pack_levels", True)

    def set_training_targets(self, *, tokenizer_updates, generator_updates,
                             precision="auto", pack_levels=True, warmup_updates=100):
        """Set TOTAL targets, not extra updates on every notebook execution.

        Existing successful tokenizers may be kept by setting tokenizer_updates
        to their current count. Changing a tokenizer after stage II requires an
        explicit new generator run, so it is never done silently.
        """
        if min(tokenizer_updates, generator_updates, warmup_updates) < 1:
            raise ValueError("Update targets and warmup must be positive.")
        if self.state["generator"]["updates"] and tokenizer_updates > self.state["tokenizer"]["updates"]:
            raise ValueError("Keep the trained tokenizer fixed when continuing its generator.")
        if precision == "auto":
            precision = "bfloat16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        if precision not in ("float32", "bfloat16"):
            raise ValueError("Use float32, bfloat16, or auto precision.")
        previous = self.plan
        reference = None if previous is None else previous.get("original_time_budget", previous)
        if reference is not None and reference.get("mode") == "updates":
            reference = None
        self.plan = dict(mode="updates", tokenizer_updates=int(tokenizer_updates),
                         generator_updates=int(generator_updates), warmup_updates=int(warmup_updates),
                         generator_precision=precision, tokenizer_precision="float32",
                         pack_levels=bool(pack_levels), original_time_budget=reference)
        self.configure_runtime()

    @torch.no_grad()
    def tokenizer_quality(self, dataset, examples=128):
        """Full-hierarchy accuracy on validation only; no updates or EMA writes."""
        self.model.eval()
        examples = min(examples, len(dataset))
        pixels, correct, exact = 0, 0, 0
        for start in range(0, examples, self.micro_batch_size):
            items = default_collate([dataset[i] for i in range(start, min(start+self.micro_batch_size, examples))])
            target = items if isinstance(items, torch.Tensor) else items[-1]
            target = target.to(self.device)
            _, reconstructions = self.model.tokenizer.quantize(self.model.tokenizer.latent(target))
            prediction = self.model.tokenizer.decode(reconstructions[-1]).argmax(1)
            matches = prediction == target
            correct += int(matches.sum())
            pixels += matches.numel()
            exact += int(matches.flatten(1).all(1).sum())
        return dict(examples=examples, pixel_accuracy=correct/pixels, exact_accuracy=exact/examples)

    def require_trained(self):
        if any(self.state[stage]["updates"] == 0 for stage in self.state):
            raise RuntimeError("Train or load both NCM stages before generation/evaluation.")

    def report(self):
        return dict(parameters=self.model.parameter_counts(), config=self.config.to_dict(),
                    budget=copy.deepcopy(self.plan), stages=copy.deepcopy(self.state),
                    batch_size=self.batch_size, micro_batch_size=self.micro_batch_size,
                    seed=self.seed, data_signature=self.signature, paper=PAPER,
                    reference_code=REFERENCE,
                    precision=dict(tokenizer="float32", generator=getattr(self.model, "generator_precision", "float32")),
                    timing_scope="data fetching + train + validation; excludes plotting/checkpoint IO",
                    comparison=("expanded optimizer-update budget; no equal-time claim"
                                if (self.plan or {}).get("mode") == "updates" else
                                "estimated same-machine baseline training time, not measured historic wall time"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format_version=self.FORMAT, config=self.config.to_dict(),
                       model_state_dict=self.model.state_dict(),
                       optimizer_state_dicts={k: v.state_dict() for k, v in self.optimizers.items()},
                       history=self.history, state=self.state, plan=self.plan, rng=self.rng,
                       data_signature=self.signature, seed=self.seed,
                       learning_rate=self.learning_rate, batch_size=self.batch_size,
                       micro_batch_size=self.micro_batch_size)
        temporary = self.path.with_suffix(".tmp.pt")
        torch.save(payload, temporary)
        temporary.replace(self.path)
        self.path.with_suffix(".budget.json").write_text(
            json.dumps(self.report(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self):
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != self.FORMAT:
            raise ValueError("Not a grid_ncm_v1 checkpoint.")
        if checkpoint["data_signature"] != self.signature:
            raise ValueError("Dataset or split changed; use the matching checkpoint.")
        if checkpoint["config"] != self.config.to_dict():
            raise ValueError("Architecture differs from this checkpoint.")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        for stage, optimizer in self.optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizer_state_dicts"][stage])
        self.history, self.state = checkpoint["history"], checkpoint["state"]
        self.plan, self.rng = checkpoint["plan"], checkpoint["rng"]
        self.seed, self.learning_rate = checkpoint["seed"], checkpoint["learning_rate"]
        self.batch_size = checkpoint["batch_size"]
        self.micro_batch_size = checkpoint["micro_batch_size"]
        self.model.tokenizer.eval()
        self.configure_runtime()
        return self

    def _batch(self, dataset, count=None):
        count = self.batch_size if count is None else count
        indices = torch.randint(len(dataset), (count,)).tolist()
        items = default_collate([dataset[i] for i in indices])
        if isinstance(items, torch.Tensor):
            return items.to(self.device), None
        if len(items) == 1:
            return items[0].to(self.device), None
        # Conditional datasets return (original coloring, colored target).
        return items[1].to(self.device), items[0].to(self.device)

    @torch.no_grad()
    def validate(self, stage, dataset, examples=128):
        # Fixed validation subset; no augmentation and no influence on training RNG.
        count = min(examples, len(dataset))
        total, accuracy, exact, used = 0., 0., 0, 0
        level_totals = np.zeros(len(self.config.scales))
        counts = [s*s for s in self.config.scales]
        denominator = sum(n**(1-self.config.alpha) for n in counts)
        level_weights = np.asarray([n**(1-self.config.alpha)/denominator for n in counts])
        self.model.eval()
        for start in range(0, count, self.micro_batch_size):
            items = default_collate([dataset[i] for i in range(start, min(start + self.micro_batch_size, count))])
            if isinstance(items, torch.Tensor):
                target, coloring = items.to(self.device), None
            elif len(items) == 1:
                target, coloring = items[0].to(self.device), None
            else:
                target, coloring = items[1].to(self.device), items[0].to(self.device)
            if stage == "tokenizer":
                loss, metrics = self.model.tokenizer.loss(target, nested=False)
                accuracy += float(metrics["accuracy"]) * len(target)
                exact += float(metrics["exact_accuracy"]) * len(target)
            else:
                losses = list(self.model.generator_losses(target, coloring, corrupt=False))
                level_totals += torch.stack(losses).float().cpu().numpy() / level_weights * len(target)
                loss = sum(losses)
            total += float(loss) * len(target)
            used += len(target)
        return dict(validation_loss=total / used,
                    validation_accuracy=accuracy / used if stage == "tokenizer" else None,
                    validation_exact_accuracy=exact/used if stage == "tokenizer" else None,
                    validation_level_ce=(level_totals/used).tolist() if stage == "generator" else None)

    def fit(self, stage, dataset, validation_dataset, *, report_every=100,
            validation_examples=128, callback=None):
        """Continue a stage to its explicit update target or legacy time cap.

        Stage time includes online validation and fetching/augmenting examples.
        Reporting and saving are outside the cap. A cap can overshoot by one
        update + one validation pass; both durations remain in the audit report.
        """
        if self.plan is None:
            raise RuntimeError("Configure the training targets first.")
        if stage not in self.state:
            raise ValueError(stage)
        if stage == "tokenizer" and self.state["generator"]["updates"]:
            raise RuntimeError("Changing a tokenizer invalidates trained generator labels.")
        if stage == "generator" and not self.state["tokenizer"]["updates"]:
            raise RuntimeError("Train/load the tokenizer first.")
        if (stage == "generator" and self.plan.get("mode") == "updates"
                and self.state["tokenizer"]["updates"] < self.plan["tokenizer_updates"]):
            raise RuntimeError("Complete the full tokenizer update target before training the generator.")
        optimizer = self.optimizers[stage]
        selected = self.model.tokenizer if stage == "tokenizer" else self.model.generator
        if stage == "generator":
            self.model.tokenizer.requires_grad_(False).eval()
        self.configure_runtime()
        updates_mode = self.plan.get("mode") == "updates"
        cap = self.plan[f"{stage}_updates" if updates_mode else f"{stage}_seconds"]
        state = self.state[stage]
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.rng["cpu"])
            if self.device.type == "cuda" and self.rng["cuda"]:
                torch.cuda.set_rng_state_all(self.rng["cuda"])
            try:
                while state["updates" if updates_mode else "seconds"] < cap:
                    synchronize(self.device)
                    begin = time.perf_counter()
                    self.model.train()
                    if stage == "generator":
                        self.model.tokenizer.eval()
                    warmup_steps = min(self.plan.get("warmup_updates", 50), max(1, int(cap)//10)) if updates_mode else 50
                    progress = (max(0., (state["updates"]-warmup_steps) / max(1, cap-warmup_steps))
                                if updates_mode else min(1., state["seconds"] / max(cap, 1e-9)))
                    warmup = min(1., (state["updates"] + 1) / warmup_steps)
                    lr = self.learning_rate * warmup * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    target, coloring = self._batch(dataset)
                    optimizer.zero_grad(set_to_none=True)
                    training_loss, ce, vq = 0., 0., 0.
                    for start in range(0, len(target), self.micro_batch_size):
                        chunk = target[start:start + self.micro_batch_size]
                        condition = None if coloring is None else coloring[start:start + len(chunk)]
                        weight = len(chunk) / len(target)
                        if stage == "tokenizer":
                            loss, metrics = self.model.tokenizer.loss(chunk)
                            (loss * weight).backward()
                            training_loss += float(loss.detach()) * weight
                            ce += float(metrics["ce"]) * weight
                            vq += float(metrics["vq"]) * weight
                        else:
                            loss = sum(self.model.generator_losses(chunk, condition))
                            (loss * weight).backward()
                            training_loss += float(loss.detach()) * weight
                    torch.nn.utils.clip_grad_norm_(selected.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    state["updates"] += 1
                    state["examples"] += len(target)
                    synchronize(self.device)
                    state["seconds"] += time.perf_counter() - begin
                    entry = dict(update=state["updates"], seconds=state["seconds"], loss=training_loss,
                                 ce=ce if stage == "tokenizer" else training_loss,
                                 vq=vq if stage == "tokenizer" else 0., learning_rate=lr)
                    finished = state["updates" if updates_mode else "seconds"] >= cap
                    reporting = state["updates"] == 1 or state["updates"] % report_every == 0 or finished
                    if reporting:
                        synchronize(self.device)
                        begin = time.perf_counter()
                        entry.update(self.validate(stage, validation_dataset, validation_examples))
                        synchronize(self.device)
                        state["seconds"] += time.perf_counter() - begin
                        entry["seconds"] = state["seconds"]
                    self.history[stage].append(entry)
                    self.rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
                                if self.device.type == "cuda" else []}
                    if reporting:
                        self.save()
                        if callback:
                            callback(self, stage)
            finally:
                self.rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
                            if self.device.type == "cuda" else []}
                self.save()
                self.model.eval()
                if callback:
                    callback(self, stage)


def train_online(run, stage, train_dataset, validation_dataset, *, report_every=100):
    """Display one live FigureWidget; update its existing traces in place."""
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from IPython.display import display

    if run.plan is None:
        raise RuntimeError("Configure training targets or load the saved run first.")
    updates_mode = run.plan.get("mode") == "updates"
    cap = run.plan[f"{stage}_updates" if updates_mode else f"{stage}_seconds"]
    progress_key = "updates" if updates_mode else "seconds"
    progress = widgets.FloatProgress(min=0, max=cap, value=min(cap, run.state[stage][progress_key]),
                                    description=stage.title(), style={"description_width": "initial"},
                                    layout=widgets.Layout(width="100%"))
    fig = go.FigureWidget(
        data=[
            go.Scatter(name="Train objective", mode="lines",
                       line=dict(color="#4969ed", width=2)),
            go.Scatter(name="Validation (full hierarchy)", mode="lines+markers",
                       line=dict(color="#e99532", width=2)),
        ],
        layout=go.Layout(
            template="plotly_white", height=390, autosize=True,
            xaxis_title="Optimizer updates", yaxis_title="Loss", yaxis_type="log",
            margin=dict(l=55, r=20, t=85, b=55),
            legend=dict(orientation="h", y=1.12),
        ),
    )

    def update(current, current_stage):
        history = current.history[current_stage]
        validation = [h for h in history if "validation_loss" in h]
        state = current.state[current_stage]
        # Widget comms update the same frontend figure; no display_data messages
        # or repeated HTML renders are emitted by this callback.
        with fig.batch_update():
            fig.data[0].x = [h["update"] for h in history]
            fig.data[0].y = [h["loss"] for h in history]
            fig.data[1].x = [h["update"] for h in validation]
            fig.data[1].y = [h["validation_loss"] for h in validation]
            fig.layout.title.text = (
                f"NCM · {current_stage} · {state['updates']:,}"
                + (f" / {int(cap):,} updates · {state['seconds']/60:.1f} min" if updates_mode else
                   f" updates · {state['seconds']/60:.1f} / {cap/60:.1f} min")
            )
        progress.value = min(cap, state[progress_key])
    update(run, stage)
    display(widgets.VBox([progress, fig], layout=widgets.Layout(width="100%")))
    run.fit(stage, train_dataset, validation_dataset, report_every=report_every, callback=update)
    progress.bar_style = "success"
