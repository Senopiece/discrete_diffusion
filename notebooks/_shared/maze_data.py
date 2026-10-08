"""Generate and evaluate binary perfect mazes on a 32x32 pixel grid."""

from typing import Iterable

import numpy as np
from numba import njit


GRID_SIZE = 32
WALL = 0
PASSAGE = 1
START = (0, 1)
GOAL = (31, 30)


def generate_maze(rng: np.random.Generator, size: int = GRID_SIZE) -> np.ndarray:
    """Carve a randomized depth-first-search maze; return 0=wall, 1=passage."""
    if size != GRID_SIZE:
        raise ValueError(f"This experiment expects size={GRID_SIZE}, got {size}.")

    maze = np.zeros((size, size), dtype=np.uint8)
    # Cell centers are spaced by two pixels, with a wider final interval so the
    # outer wall stays one pixel thick on all four sides of the 32x32 canvas.
    cell_rows = (size - 2) // 2
    cell_cols = (size - 2) // 2
    row_positions = [1 + 2 * i for i in range(cell_rows - 1)] + [size - 2]
    col_positions = [1 + 2 * i for i in range(cell_cols - 1)] + [size - 2]
    visited = np.zeros((cell_rows, cell_cols), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    maze[row_positions[0], col_positions[0]] = PASSAGE

    while stack:
        row, col = stack[-1]
        candidates = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < cell_rows and 0 <= nc < cell_cols and not visited[nr, nc]:
                candidates.append((nr, nc, dr, dc))
        if not candidates:
            stack.pop()
            continue

        nr, nc, dr, dc = candidates[int(rng.integers(len(candidates)))]
        visited[nr, nc] = True
        r, c = row_positions[row], col_positions[col]
        next_r, next_c = row_positions[nr], col_positions[nc]
        distance = max(abs(next_r - r), abs(next_c - c))
        for step in range(1, distance + 1):
            maze[r + dr * step, c + dc * step] = PASSAGE
        stack.append((nr, nc))

    maze[START] = PASSAGE
    maze[GOAL] = PASSAGE
    return maze


def generate_dataset(count: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic uint8 array shaped (count, 32, 32)."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return np.stack([generate_maze(rng) for _ in range(count)])


@njit(cache=True)
def _is_solvable_numba(maze: np.ndarray) -> bool:
    """Numba-compiled breadth-first search on the fixed 32x32 grid."""
    queue = np.empty((GRID_SIZE * GRID_SIZE, 2), dtype=np.int64)
    seen = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    head, tail = 0, 1
    queue[0, 0], queue[0, 1] = START[0], START[1]
    seen[START[0], START[1]] = 1
    drs = np.array((-1, 1, 0, 0))
    dcs = np.array((0, 0, -1, 1))

    while head < tail:
        row, col = queue[head, 0], queue[head, 1]
        head += 1
        if row == GOAL[0] and col == GOAL[1]:
            return True
        for direction in range(4):
            nr, nc = row + drs[direction], col + dcs[direction]
            if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                    and seen[nr, nc] == 0 and maze[nr, nc] == PASSAGE):
                seen[nr, nc] = 1
                queue[tail, 0], queue[tail, 1] = nr, nc
                tail += 1
    return False


def is_solvable(maze: np.ndarray) -> bool:
    """Check four-neighbor passage connectivity from the fixed entrance to exit."""
    if maze.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"Expected shape {(GRID_SIZE, GRID_SIZE)}, got {maze.shape}.")
    if maze[START] != PASSAGE or maze[GOAL] != PASSAGE:
        return False
    return bool(_is_solvable_numba(np.ascontiguousarray(maze, dtype=np.uint8)))


def solvability_rate(mazes: Iterable[np.ndarray]) -> float:
    """Fraction of generated mazes that connect entrance and exit."""
    items = list(mazes)
    if not items:
        raise ValueError("mazes cannot be empty")
    return float(np.mean([is_solvable(maze) for maze in items]))

