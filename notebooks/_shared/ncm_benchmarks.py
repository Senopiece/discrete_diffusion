"""Common evaluation and plotting for the maze and Queens NCM notebooks."""

from collections import deque
import csv
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from .maze_data import maze_endpoints
from .ncm_training import file_sha256, synchronize
from .queens_data import EMPTY, solution_checks


PLOT_CONFIG = {"responsive": True, "displaylogo": False}
PALETTE = ["#B8A5E8", "#F0B48B", "#92C9E8", "#A9D4A4", "#E8CD83", "#ECA6BB", "#9FB9D4", "#F1F3F5"]
COLOR_SCALE = [[p, color] for i, color in enumerate(PALETTE)
               for p in (i / len(PALETTE), (i + 1) / len(PALETTE))]
COLORS = ["#4969ed", "#e99532", "#169a87", "#a657b5"]


def progress_bar(title, total):
    import ipywidgets as widgets
    from IPython.display import display
    progress = widgets.IntProgress(value=0, max=max(1, total), description=title,
                                  style={"description_width": "initial"},
                                  layout=widgets.Layout(width="100%"))
    display(progress)
    return progress


def board_panels(boards, titles, title, *, colorings=None, queens=False, queen_masks=None, columns=6):
    boards = list(boards)
    if not boards:
        raise ValueError("No boards to show.")
    columns = min(columns, len(boards))
    rows = math.ceil(len(boards) / columns)
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=titles,
                        horizontal_spacing=.025, vertical_spacing=min(.12, .3 / rows))
    for i, board in enumerate(boards):
        row, col = i // columns + 1, i % columns + 1
        coloring = None if colorings is None else np.asarray(colorings[i])
        fig.add_trace(go.Heatmap(z=np.asarray(board), zmin=-.5 if coloring is not None else 0,
                                zmax=7.5 if coloring is not None else 1,
                                colorscale=COLOR_SCALE if coloring is not None else [[0, "#111827"], [1, "#fff"]],
                                showscale=False, xgap=1 if coloring is not None else 0,
                                ygap=1 if coloring is not None else 0, hoverinfo="skip"), row=row, col=col)
        if coloring is not None:
            xs, ys = [], []
            for r in range(len(coloring)):
                for c in range(len(coloring)):
                    if c + 1 < len(coloring) and coloring[r, c] != coloring[r, c + 1]:
                        xs += [c + .5, c + .5, None]
                        ys += [r - .5, r + .5, None]
                    if r + 1 < len(coloring) and coloring[r, c] != coloring[r + 1, c]:
                        xs += [c - .5, c + .5, None]
                        ys += [r + .5, r + .5, None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#64748b", width=1.4),
                                     showlegend=False, hoverinfo="skip"), row=row, col=col)
            if queens:
                occupied = np.asarray(board) != EMPTY if queen_masks is None else np.asarray(queen_masks[i])
                qr, qc = np.where(occupied)
                fig.add_trace(go.Scatter(x=qc, y=qr, mode="text", text=[chr(0x265b)] * len(qr),
                                         textfont=dict(size=25, color="#172033"),
                                         showlegend=False, hoverinfo="skip"), row=row, col=col)
        axis = fig.get_subplot(row, col).xaxis.plotly_name.replace("axis", "")
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, constrain="domain", row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed",
                         scaleanchor=axis, scaleratio=1, constrain="domain", row=row, col=col)
    fig.update_layout(template="plotly_white", title=dict(text=title, font_size=21),
                      autosize=True, height=rows * 230 + 90,
                      margin=dict(l=15, r=15, t=85, b=20), font=dict(family="Arial, sans-serif", size=15))
    return fig


def maze_metrics(maze):
    """Same neighbors, endpoints, tie-breaking and failed-search semantics as FMH."""
    white = np.asarray(maze) == 1
    size = len(white)
    start, goal = maze_endpoints(size)

    def neighbors(cell):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and white[nr, nc]:
                yield nr, nc

    distances = {}
    if white[start]:
        distances[start] = 0
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for nxt in neighbors(cell):
                if nxt not in distances:
                    distances[nxt] = distances[cell] + 1
                    queue.append(nxt)
    length = float(distances.get(goal, np.nan))
    expansions = 0
    if white[start]:
        heuristic = lambda cell: abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
        frontier, costs, closed = [(heuristic(start), 0, start)], {start: 0}, set()
        while frontier:
            _, cost, cell = heapq.heappop(frontier)
            if cell in closed or cost != costs.get(cell):
                continue
            closed.add(cell)
            expansions += 1
            if cell == goal:
                break
            for nxt in neighbors(cell):
                if cost + 1 < costs.get(nxt, float("inf")):
                    costs[nxt] = cost + 1
                    heapq.heappush(frontier, (cost + 1 + heuristic(nxt), cost + 1, nxt))
    unseen = set(map(tuple, np.argwhere(white)))
    isolated = 0
    while unseen:
        first = unseen.pop()
        connected_to_start = first in distances
        queue = [first]
        while queue:
            cell = queue.pop()
            for nxt in neighbors(cell):
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
        isolated += not connected_to_start
    return dict(solvable=bool(np.isfinite(length)), shortest_solution_length=length,
                astar_expansions=expansions,
                reachable_white_percent=100 * len(distances) / max(1, int(white.sum())),
                isolated_components=int(isolated))


def evaluate_mazes(run, *, sizes, count, batch_size, temperature, artifact_dir,
                    seed=7):
    run.require_trained()
    run.save()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    signature = dict(checkpoint_sha256=file_sha256(run.path), sizes=list(sizes), count=count,
                     batch_size=batch_size, temperature=temperature, seed=seed, version="ncm_maze_eval_v2")
    metadata_path = artifact_dir / "evaluation.json"
    arrays_path = artifact_dir / "evaluation.npz"
    if metadata_path.exists() and arrays_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(arrays_path, allow_pickle=False) as arrays:
                results = {int(size): {name: arrays[key].copy() for name, key in names.items()}
                           for size, names in metadata["keys"].items()}
            return results, metadata
    results, timings, keys = {}, {}, {}
    calls_per_batch = len(run.config.scales)
    progress = progress_bar("Generate mazes", len(sizes) * math.ceil(count / batch_size) * calls_per_batch)
    def advance(*_):
        progress.value += 1
    for size in sizes:
        synchronize(run.device)
        begin = time.perf_counter()
        batches = []
        for start in range(0, count, batch_size):
            batch = run.model.sample(min(batch_size, count - start), size=size,
                                     temperature=temperature, seed=seed + size * 100003 + start,
                                     progress=advance)
            batches.append(batch.numpy().astype(np.uint8))
        synchronize(run.device)
        timings[str(size)] = {"NCM": time.perf_counter() - begin}
        results[size] = {"NCM": np.concatenate(batches)}
    progress.bar_style = "success"
    arrays = {}
    for size, groups in results.items():
        keys[str(size)] = {}
        for i, (name, values) in enumerate(groups.items()):
            key = f"size_{size}_method_{i}"
            arrays[key] = values
            keys[str(size)][name] = key
    np.savez_compressed(arrays_path, **arrays)
    metadata = dict(signature=signature, keys=keys, generation_seconds=timings,
                    ncm_generator_evaluations=len(run.config.scales), ncm_decoder_evaluations=1,
                    training=run.report())
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results, metadata


def plot_maze_metrics(results, artifact_dir):
    artifact_dir = Path(artifact_dir)
    rows = []
    progress = progress_bar("Search metrics", sum(len(x) for groups in results.values() for x in groups.values()))
    for size, groups in results.items():
        for name, mazes in groups.items():
            for i, maze in enumerate(mazes):
                rows.append(dict(size=size, method=name, sample=i, **maze_metrics(maze)))
                progress.value += 1
    progress.bar_style = "success"
    with (artifact_dir / "maze_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = [("solvable", "Solvability", "Solved (%)"),
               ("shortest_solution_length", "Shortest solution · solved mazes only", "Grid moves"),
               ("astar_expansions", "A* expansions · all mazes", "Expanded cells"),
               ("reachable_white_percent", "Passage cells reachable from entrance", "Reachable (%)"),
               ("isolated_components", "Passage components isolated from entrance", "Components")]
    figures = []
    for metric, title, axis in metrics:
        fig = make_subplots(rows=1, cols=len(results), subplot_titles=[f"{size} × {size}" for size in results])
        for col, (size, groups) in enumerate(results.items(), 1):
            for j, name in enumerate(groups):
                values = np.array([row[metric] for row in rows if row["size"] == size and row["method"] == name], dtype=float)
                values = values[np.isfinite(values)]
                mean = float(values.mean()) if len(values) else np.nan
                if metric == "solvable":
                    n = len(values)
                    # Wilson interval, appropriate even for 0/n or n/n successes.
                    p, z = mean, 1.96
                    center = (p + z*z/(2*n))/(1+z*z/n)
                    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                    fig.add_trace(go.Bar(x=[name], y=[100*p], marker_color=COLORS[j % len(COLORS)],
                                         error_y=dict(type="data", symmetric=False,
                                                      array=[100*(center+half-p)], arrayminus=[100*(p-center+half)]),
                                         name=name, showlegend=col == 1), row=1, col=col)
                else:
                    fig.add_trace(go.Box(y=values, x=[name]*len(values), name=name, boxpoints=False,
                                         line_color=COLORS[j % len(COLORS)], showlegend=col == 1), row=1, col=col)
                    fig.add_trace(go.Scatter(x=[name], y=[mean], mode="markers", showlegend=False,
                                             marker=dict(symbol="line-ew", size=28, color="#172033", line_width=2),
                                             hovertemplate="Mean: %{y:.3f}<extra></extra>"), row=1, col=col)
            fig.update_yaxes(title_text=axis, row=1, col=col)
        fig.update_layout(template="plotly_white", title=title, height=440, autosize=True,
                          margin=dict(l=60, r=20, t=85, b=90), showlegend=False)
        fig.write_html(artifact_dir / f"{metric}.html", include_plotlyjs="cdn")
        fig.show(config=PLOT_CONFIG)
        figures.append(fig)
    return rows, figures


def evaluate_queens(run, puzzles, indices, *, attempts=32, batch_size=128, temperature=1., seed=10017,
                    artifact_dir):
    run.require_trained()
    run.save()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(indices, dtype=np.int64)
    signature = dict(checkpoint_sha256=file_sha256(run.path), indices=indices.tolist(),
                     dataset_sha256=puzzles["metadata"]["data_sha256"], attempts=attempts,
                     batch_size=batch_size, temperature=temperature, seed=seed,
                     version="ncm_queens_eval_v1")
    metadata_path, arrays_path = artifact_dir / "evaluation.json", artifact_dir / "evaluation.npz"
    if metadata_path.exists() and arrays_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(arrays_path, allow_pickle=False) as arrays:
                return arrays["successes"].copy(), arrays["tokens"].copy(), metadata
    boards = np.repeat(puzzles["regions"][indices], attempts, axis=0)
    batches = []
    progress = progress_bar("Solve Queens", math.ceil(len(boards) / batch_size))
    synchronize(run.device)
    begin = time.perf_counter()
    for start in range(0, len(boards), batch_size):
        tokens = run.model.sample(coloring=boards[start:start + batch_size], temperature=temperature, seed=seed + start)
        batches.append(tokens.numpy().astype(np.uint8))
        progress.value += 1
    synchronize(run.device)
    elapsed = time.perf_counter() - begin
    tokens = np.concatenate(batches)
    solved = solution_checks(boards, tokens)["solved"].reshape(len(indices), attempts)
    tokens = tokens.reshape(len(indices), attempts, 7, 7)
    progress.bar_style = "success"
    np.savez_compressed(arrays_path, indices=indices, successes=solved, tokens=tokens)
    metadata = dict(signature=signature, generation_seconds=elapsed,
                    generator_evaluations=len(run.config.scales), decoder_evaluations=1,
                    mean_solvability=float(solved.mean()), training=run.report())
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (artifact_dir / "solvability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["puzzle_index", "puzzle_id", "source", "successes", "attempts", "probability"])
        for i, index in enumerate(indices):
            writer.writerow([index, puzzles["fingerprints"][index], puzzles["source_names"][index],
                             int(solved[i].sum()), attempts, float(solved[i].mean())])
    return solved, tokens, metadata


def retry_curves(successes):
    """Empirical pass@k averaged over all k-subsets of the observed attempts."""
    successes = np.asarray(successes, dtype=bool)
    n = successes.shape[1]
    counts = successes.sum(1)
    survival = np.ones(len(counts), dtype=float)
    curves = [1 - survival]
    for k in range(1, n + 1):
        survival *= np.maximum(0, n - counts - k + 1) / (n - k + 1)
        curves.append(1 - survival.copy())
    return np.stack(curves, axis=1)


def plot_queens_retries(groups, *, seed=17, bootstrap=2000):
    fig = go.Figure()
    rng = np.random.default_rng(seed)
    n_puzzles = len(next(iter(groups.values())))
    bootstrap_indices = rng.integers(n_puzzles, size=(bootstrap, n_puzzles))
    for j, (name, successes) in enumerate(groups.items()):
        curves = retry_curves(successes)
        k = np.arange(curves.shape[1])
        low, high = np.quantile(curves[bootstrap_indices].mean(1), [.025, .975], axis=0)
        color = COLORS[j % len(COLORS)]
        rgb = ",".join(str(int(color[i:i+2], 16)) for i in (1, 3, 5))
        fig.add_scatter(x=k, y=100*low, line_width=0, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=k, y=100*high, line_width=0, fill="tonexty",
                        fillcolor=f"rgba({rgb},.12)", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=k, y=100*curves.mean(0), name=name, line=dict(color=color, width=3))
        p = np.asarray(successes).mean()
        fig.add_scatter(x=k, y=100*(1-(1-p)**k), name=f"{name} · from mean p",
                        line=dict(color=color, width=1.5, dash="dot"))
    fig.update_layout(template="plotly_white", title="At least one solution vs attempts",
                      xaxis_title="Independent attempts", yaxis_title="Solved puzzles (%)",
                      yaxis_range=[0, 100], height=480, autosize=True,
                      legend=dict(orientation="h", y=1.14), margin=dict(t=105, b=60))
    return fig
