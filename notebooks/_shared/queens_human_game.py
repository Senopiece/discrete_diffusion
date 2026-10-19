"""Notebook Queens game and append-only human difficulty observations."""

import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
import uuid

import anywidget
import numpy as np
import traitlets

from .queens_data import EMPTY, solution_checks


HUMAN_GAME_FIELDS = [
    "game_id", "dataset_sha256", "puzzle_id", "dataset_index", "source", "source_id",
    "player_id", "started_at_utc", "completed_at_utc", "elapsed_seconds",
    "difficulty", "actions", "undos", "notebook", "split_seed",
]


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_human_games(path):
    """Read completed, explicitly rated games; absence is an empty study."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HUMAN_GAME_FIELDS:
            raise ValueError(f"Unexpected human-game CSV columns in {path}.")
        return list(reader)


def append_human_game(path, record):
    """Serialize writes from separate notebook kernels and deduplicate game IDs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + 5
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("CSV is busy. Please press Save again in a moment.")
            time.sleep(0.05)
    try:
        os.close(descriptor)
        records = read_human_games(path)
        if any(row["game_id"] == record["game_id"] for row in records):
            return len(records)
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HUMAN_GAME_FIELDS)
            if not path.stat().st_size:
                writer.writeheader()
            writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
        return len(records) + 1
    finally:
        lock_path.unlink(missing_ok=True)


class QueensHumanGame(anywidget.AnyWidget):
    """Uniform test-puzzle draws, browser interactions, and validated CSV saving.

    Ground-truth queen positions and model success rates are never sent to the
    browser. A completed placement is checked against Queens rules in Python.
    """

    _esm = Path(__file__).with_name("queens_human_game.js")
    _css = Path(__file__).with_name("queens_human_game.css")
    puzzle = traitlets.Dict().tag(sync=True)
    feedback = traitlets.Dict().tag(sync=True)
    player = traitlets.Unicode("local").tag(sync=True)
    total_saved = traitlets.Int(0).tag(sync=True)

    def __init__(self, puzzles, test_indices, csv_path, *, player="local",
                 notebook="", split_seed=17, **kwargs):
        super().__init__(**kwargs)
        indices = np.asarray(test_indices, dtype=np.int64)
        if (
            indices.ndim != 1 or not len(indices)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(puzzles["regions"]))
        ):
            raise ValueError("Expected distinct valid test-puzzle indices.")
        self._puzzles = puzzles
        self._indices = indices
        self._csv_path = Path(csv_path)
        self._notebook = str(notebook)
        self._split_seed = int(split_seed)
        self._rng = np.random.default_rng()
        self._started_at = None
        self._completed = None
        self._saved = False
        self.player = str(player).strip() or "local"
        self.total_saved = len(read_human_games(self._csv_path))
        self.on_msg(self._handle_message)
        self._new_game()

    def _new_game(self):
        index = int(self._rng.choice(self._indices))
        self._started_at = None
        self._completed = None
        self._saved = False
        self.puzzle = dict(
            game_id=uuid.uuid4().hex,
            dataset_index=index,
            puzzle_id=str(self._puzzles["fingerprints"][index]),
            board=self._puzzles["regions"][index].tolist(),
        )
        self.feedback = dict(game_id=self.puzzle["game_id"], kind="ready", text="")

    def _reply(self, kind, text):
        self.feedback = dict(
            game_id=self.puzzle["game_id"], kind=kind, text=text,
            sequence=uuid.uuid4().hex,
        )

    def _check_finish(self, message):
        board = self._puzzles["regions"][self.puzzle["dataset_index"]]
        mask = np.asarray(message["queens"])
        if mask.shape != board.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("Invalid queen placement.")
        tokens = np.where(mask.astype(bool), board, EMPTY)
        if not bool(solution_checks(board, tokens)["solved"]):
            raise ValueError("The selected queens do not yet satisfy all rules.")
        seconds = float(message["elapsed_seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid completion time.")
        self._completed = dict(
            elapsed_seconds=seconds, completed_at_utc=_utc_now(),
            actions=max(0, int(message.get("actions", 0))),
            undos=max(0, int(message.get("undos", 0))),
        )

    def _handle_message(self, widget, message, buffers):
        if message.get("type") == "new_game":
            # Each draw is independent and uniform over the entire test split.
            self.player = str(message.get("player", self.player)).strip() or "local"
            self._new_game()
            return
        if message.get("game_id") != self.puzzle["game_id"]:
            return  # Ignore late messages from a previous board.
        try:
            kind = message.get("type")
            if kind == "start":
                if self._started_at is None:
                    self._started_at = _utc_now()
                self.player = str(message.get("player", self.player)).strip() or "local"
            elif kind == "resume" and not self._saved:
                self._completed = None
            elif kind == "solved" and not self._saved:
                self._check_finish(message)
                self._reply("solved", "Решено. Оцени сложность от 0 до 10.")
            elif kind == "save":
                if self._saved:
                    self._reply("saved", "Уже сохранено. Можно открыть следующую раскраску.")
                    return
                if self._completed is None or self._started_at is None:
                    raise ValueError("First complete the puzzle.")
                difficulty = float(message["difficulty"])
                if not math.isfinite(difficulty) or not 0 <= difficulty <= 10:
                    raise ValueError("Choose a difficulty from 0 to 10.")
                index = self.puzzle["dataset_index"]
                record = dict(
                    game_id=self.puzzle["game_id"],
                    dataset_sha256=self._puzzles["metadata"]["data_sha256"],
                    puzzle_id=self.puzzle["puzzle_id"], dataset_index=index,
                    source=str(self._puzzles["source_names"][index]),
                    source_id=str(self._puzzles["source_ids"][index]),
                    player_id=self.player, started_at_utc=self._started_at,
                    difficulty=difficulty, notebook=self._notebook,
                    split_seed=self._split_seed, **self._completed,
                )
                self.total_saved = append_human_game(self._csv_path, record)
                self._saved = True
                self._reply("saved", "Сохранено. Можно открыть следующую раскраску.")
        except (KeyError, TypeError, ValueError, OSError, TimeoutError) as error:
            self._reply("error", str(error))
