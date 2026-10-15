"""Collect ready-made 7x7 Queens pairs from three sources. Never solve puzzles.

Run: uv run python scripts/collect_queens7.py
Source downloads are cached under artifacts/queens7/sources.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
from _shared.queens_data import normalize_regions, puzzle_fingerprint, solution_checks

CACHE = PROJECT_ROOT / "artifacts" / "queens7" / "sources"
OUTPUT = PROJECT_ROOT / "data" / "queens7"
AMAZE_REVISION = "ba5628fad345c2032d64dfc5e31e416492387a67"
QUEENDOM_REVISION = "e53a8c3ce9ad7ca47f24ecb16c07bab23e814350"
QUEENS_GAME_BUNDLES = range(1, 11)


def download(url, filename):
    path = CACHE / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "queens7-dataset-collector"})
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return path


def collect():
    CACHE.mkdir(parents=True, exist_ok=True)
    records = {}
    counts = Counter()
    snapshots = []
    source_counts = Counter()

    def snapshot(url, path):
        snapshots.append({
            "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })

    def add(regions, queens, source):
        board = normalize_regions(regions)
        coordinates = np.asarray(queens, dtype=np.int64)
        if board.shape != (7, 7) or coordinates.shape != (7, 2):
            counts["malformed"] += 1
            return
        if len(np.unique(board)) != 7 or np.any(coordinates < 0) or np.any(coordinates >= 7):
            counts["malformed"] += 1
            return
        mask = np.zeros((7, 7), dtype=bool)
        mask[coordinates[:, 0], coordinates[:, 1]] = True
        tokens = np.where(mask, board, 7)
        # Only verify the supplied answer's constraints. No alternative is searched.
        if not bool(solution_checks(board, tokens)["solved"]):
            counts["invalid_supplied_answer"] += 1
            return
        fingerprint = puzzle_fingerprint(board)
        counts["ready_7x7_pairs"] += 1
        if fingerprint in records:
            records[fingerprint]["sources"].append(source)
            counts["duplicates_up_to_symmetry"] += 1
            return
        records[fingerprint] = {
            "id": f"q7_{fingerprint[:16]}",
            "fingerprint": fingerprint,
            "regions": board.tolist(),
            "queens": coordinates.tolist(),
            "unique_solution": "source_claim",
            "sources": [source],
        }
        source_counts[source["name"]] += 1

    amaze_url = (
        f"https://huggingface.co/datasets/piekenius123/Amaze/resolve/{AMAZE_REVISION}/"
        "queen/queen-dataset/queen_dataset_test.parquet"
    )
    path = download(amaze_url, f"amaze_{AMAZE_REVISION}.parquet")
    snapshot(amaze_url, path)
    rows = pq.read_table(
        path, columns=["id", "n", "sample_json"], filters=[("n", "=", 7)]
    ).to_pylist()
    for row in rows:
        pair = json.loads(row["sample_json"])
        add(pair["regions"], pair["queens"], {
            "name": "AMAZE",
            "id": row["id"],
            "url": "https://huggingface.co/datasets/piekenius123/Amaze",
            "file": "queen/queen-dataset/queen_dataset_test.parquet",
            "revision": AMAZE_REVISION,
            "uniqueness": "published unique-solution generation pipeline",
            "label": "sample_json.queens",
        })

    for number in QUEENS_GAME_BUNDLES:
        url = f"https://queens.game/map-bundles/bundle-{number}.json"
        path = download(url, f"queens_game/bundle-{number}.json")
        snapshot(url, path)
        for key, pair in json.loads(path.read_bytes()).items():
            if pair.get("caseNumber") != 7 or pair.get("solutionCount") != 1:
                continue
            columns = pair.get("solvingCombination")
            if not isinstance(columns, list) or len(columns) != 7:
                counts["missing_supplied_answer"] += 1
                continue
            add(pair["colorGrid"], list(enumerate(columns)), {
                "name": "Queens.game",
                "id": key,
                "url": url,
                "solution_count": 1,
                "uniqueness": "published solutionCount == 1",
                "label": "solvingCombination: column per row, zero-based",
            })

    queendom_url = f"https://codeload.github.com/bonesmoses/queendom/zip/{QUEENDOM_REVISION}"
    path = download(queendom_url, f"bonesmoses_queendom_{QUEENDOM_REVISION}.zip")
    snapshot(queendom_url, path)
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith("/test/fixtures.json"))
        for pair in json.loads(archive.read(name)):
            if pair["size"] != 7:
                continue
            add(pair["regions"], pair["solution"], {
                "name": "Queendom",
                "id": pair["name"],
                "url": f"https://github.com/bonesmoses/queendom/blob/{QUEENDOM_REVISION}/test/fixtures.json",
                "revision": QUEENDOM_REVISION,
                "license": "MIT",
                "uniqueness": "source fixture from its logic-solvable puzzle collection",
                "label": "solution: [row, column] pairs, zero-based",
            })

    ordered = sorted(records.values(), key=lambda record: record["id"])
    payload = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n" for record in ordered
    ).encode("utf-8")
    manifest = {
        "version": "queens7_multisource_ready_labels_v1",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "board_size": 7,
        "counts": {**dict(counts), "accepted": len(ordered)},
        "source_counts": dict(source_counts),
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "rules": "one_per_row_column_region_no_diagonal_touching",
        "uniqueness": "source claims only; no solver or uniqueness search was run",
        "deduplication": "D4 rotations/reflections and arbitrary color relabeling",
        "snapshots": snapshots,
        "notes": [
            "All solution coordinates were supplied by the sources.",
            "Only the format and validity of each supplied answer were checked.",
            "AMAZE pairs originate in its published test split; our split is independent.",
            "Source snapshots are cached; numerical records are provided with per-pair provenance.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "puzzles.jsonl").write_bytes(payload)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "sources": manifest["source_counts"]}, indent=2))


if __name__ == "__main__":
    collect()

