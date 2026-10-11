"""Render an adaptive sampling history to a standalone, seekable MP4."""
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def _prefix(xs, ys, end):
    """Reveal a polyline up to a fractional array index."""
    end = float(np.clip(end, 0, len(xs) - 1))
    index = int(end)
    x, y = xs[: index + 1], ys[: index + 1]
    fraction = end - index
    if fraction > 0 and index + 1 < len(xs):
        x = np.append(x, xs[index] + fraction * (xs[index + 1] - xs[index]))
        y = np.append(y, ys[index] + fraction * (ys[index + 1] - ys[index]))
    return x, y


def render_adaptive_video(
    history,
    output_path,
    *,
    fps=24,
    width=1280,
    height=720,
    trace_speed_pixels_per_second=520,
    hold_seconds=1.0,
):
    """Stream Agg-rendered RGB frames into H.264 MP4; no browser or GUI required.

    History is produced by adaptive_entropy_sample(..., visualize=True).
    Only one RGB frame is held in memory at a time.
    """
    if not history:
        raise ValueError("Sampling history must not be empty.")
    if fps <= 0 or width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Use positive fps and positive even video dimensions.")
    if trace_speed_pixels_per_second <= 0 or hold_seconds < 0:
        raise ValueError("Trace speed must be positive; hold duration cannot be negative.")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "font.size": 19,
        "text.color": "#172033",
        "axes.labelcolor": "#69758a",
        "xtick.color": "#8b95a6",
        "ytick.color": "#8b95a6",
    }):
        return _render(history, output_path, fps, width, height,
                       trace_speed_pixels_per_second, hold_seconds)


def _render(history, path, fps, width, height, trace_speed_pixels_per_second, hold_seconds):
    blue, red, muted = "#4263eb", "#d94c55", "#747d8d"
    ln2 = float(np.log(2.0))
    total = len(history) - 1
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes([0.09, 0.19, 0.57, 0.59])
    sample_axes = figure.add_axes([0.72, 0.23, 0.24, 0.51])

    heading = figure.text(0.09, 0.95, "", size=28, weight="semibold", color="#263248")
    sample_axes.set_title("Iteration result", fontsize=23, color="#4d5a70", pad=22)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, ln2 * 1.06)
    axes.set_xlabel(r"Candidate time $t_1$", labelpad=12, fontsize=22)
    axes.set_ylabel("Entropy (nats)", labelpad=12, fontsize=22)
    axes.grid(color="#e3e8f0", linewidth=1.0)
    axes.set_axisbelow(True)
    axes.tick_params(length=0, pad=10, labelsize=19)
    for edge in ("top", "right", "left"):
        axes.spines[edge].set_visible(False)
    axes.spines["bottom"].set_color("#dfe4ec")

    target, = axes.plot([], [], color=muted, linestyle=(0, (5, 4)), linewidth=2.2)
    estimate, = axes.plot([], [], color=blue, linewidth=3.6, solid_capstyle="round")
    continuation, = axes.plot([], [], color=blue, alpha=0.5,
                              linestyle=(0, (4, 3)), linewidth=2.2)
    accepted, = axes.plot([], [], color=red, linewidth=2.8,
                         solid_capstyle="round", zorder=4)
    current = axes.axvline(0, color=red, alpha=0.7, linewidth=1.2,
                           linestyle=(0, (2, 3)), label=r"Frontier $t_0$")
    image = sample_axes.imshow(history[0]["after_image"], cmap="gray", vmin=0, vmax=1,
                               interpolation="nearest")
    sample_axes.set_axis_off()
    figure.legend([
        Line2D([], [], color=muted, linestyle="--", linewidth=2.2),
        Line2D([], [], color=blue, linewidth=3.6),
        Line2D([], [], color=blue, linestyle="--", alpha=0.65, linewidth=2.2),
        Line2D([], [], color=red, linewidth=2.8),
        Line2D([], [], color=red, linestyle=(0, (2, 3)), linewidth=1.8),
    ], ["Local target", "Evaluated entropy", "Uncomputed continuation", "Previous steps", r"Frontier $t_0$"],
        loc="upper left", bbox_to_anchor=(0.08, 0.875), ncol=3, frameon=False,
        fontsize=18, labelcolor="#36445b", columnspacing=1.7, handlelength=2.0,
        handletextpad=0.5, borderaxespad=0)

    arrows = []
    arrow_signature = None

    def update(record, solid_end, *, show_stop=False, tail_end=None, final=False):
        nonlocal arrows, arrow_signature
        iteration, t0 = record["iteration"], record["t0"]
        xs, hs = record["scan_t"], record["scan_h"]
        stop = record["stop_index"]
        target.set_data([t0, 1], [record["h0"], 0])
        if final:
            estimate.set_data([], [])
        else:
            estimate.set_data(*_prefix(xs, hs, solid_end))
        current.set_xdata([t0, t0])
        current.set_visible(not final)
        past = [(0.0, ln2)] + [point for point in record["prior_stops"] if point[0] <= t0 + 1e-12]
        if final:
            past.append((record["t_next"], record["stop_h"]))
        accepted.set_data([point[0] for point in past], [point[1] for point in past])
        path_points = past
        signature = tuple((round(float(x), 7), round(float(y), 7)) for x, y in path_points)
        if signature != arrow_signature:
            for arrow in arrows:
                arrow.remove()
            arrows = []
            for (x0, y0), (x1, y1) in zip(path_points, path_points[1:]):
                arrows.append(axes.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=red, lw=2.8,
                                    mutation_scale=17, shrinkA=0, shrinkB=0),
                    zorder=5,
                ))
            arrow_signature = signature
        if not final and tail_end is not None and stop < len(xs) - 1:
            continuation.set_data(*_prefix(xs[stop:], hs[stop:], tail_end))
        else:
            continuation.set_data([], [])
        image.set_data(record["after_image"] if final else record["before_image"])
        heading.set_text(f"Iteration {iteration} / {total}")

    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-crf", "20", "-preset", "fast", "-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )
    writer.send(None)

    def send_frame(repeat=1):
        canvas.draw()
        rgb = np.ascontiguousarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
        for _ in range(repeat):
            writer.send(rgb)

    try:
        update(history[0], 0)
        send_frame(1)
        for record in history[1:]:
            stop = record["stop_index"]
            xs, hs = record["scan_t"], record["scan_h"]
            points_px = axes.transData.transform(np.column_stack((xs[: stop + 1], hs[: stop + 1])))
            trace_length = float(np.linalg.norm(np.diff(points_px, axis=0), axis=1).sum())
            trace_frames = max(1, round(fps * trace_length / trace_speed_pixels_per_second))
            for end in np.linspace(0, stop, trace_frames + 1)[1:]:
                update(record, end)
                send_frame()
            tail_length = len(record["scan_t"]) - 1 - stop
            update(record, stop, show_stop=True,
                   tail_end=tail_length if tail_length > 0 else None)
            send_frame(max(1, round(fps * hold_seconds)))
        last = history[-1]
        update(last, last["stop_index"], show_stop=True, final=True)
        send_frame(max(1, round(fps * 2)))
    finally:
        writer.close()
        figure.clear()
    return path
