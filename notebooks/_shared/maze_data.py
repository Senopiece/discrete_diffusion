"""Generate and evaluate binary perfect mazes on configurable square grids."""

from typing import Iterable

import numpy as np
from numba import njit


GRID_SIZE = 32  # Default size retained for existing notebook callers.
WALL = 0
PASSAGE = 1


def maze_endpoints(size: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return fixed top-left entrance and bottom-right exit for a square grid."""
    if size < 6:
        raise ValueError(f"Maze size must be an integer >= 6, got {size}.")
    return (0, 1), (size - 1, size - 2)


START, GOAL = maze_endpoints(GRID_SIZE)


def generate_maze(rng: np.random.Generator, size: int = GRID_SIZE) -> np.ndarray:
    """Carve a randomized depth-first-search maze; return 0=wall, 1=passage."""
    start, goal = maze_endpoints(size)

    maze = np.zeros((size, size), dtype=np.uint8)
    # Cell centers are spaced by two pixels, with a wider final interval so the
    # outer wall stays one pixel thick on the square canvas.
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

    maze[start] = PASSAGE
    maze[goal] = PASSAGE
    return maze


def generate_dataset(
    count: int, seed: int = 0, size: int = GRID_SIZE
) -> np.ndarray:
    """Return deterministic mazes with shape (count, size, size)."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return np.stack([generate_maze(rng, size=size) for _ in range(count)])


@njit(cache=True)
def _is_solvable_numba(
    maze: np.ndarray, start_row: int, start_col: int, goal_row: int, goal_col: int
) -> bool:
    """Numba-compiled breadth-first search on a square grid of any supported size."""
    size = maze.shape[0]
    queue = np.empty((size * size, 2), dtype=np.int64)
    seen = np.zeros((size, size), dtype=np.uint8)
    head, tail = 0, 1
    queue[0, 0], queue[0, 1] = start_row, start_col
    seen[start_row, start_col] = 1
    drs = np.array((-1, 1, 0, 0))
    dcs = np.array((0, 0, -1, 1))

    while head < tail:
        row, col = queue[head, 0], queue[head, 1]
        head += 1
        if row == goal_row and col == goal_col:
            return True
        for direction in range(4):
            nr, nc = row + drs[direction], col + dcs[direction]
            if (0 <= nr < size and 0 <= nc < size
                    and seen[nr, nc] == 0 and maze[nr, nc] == PASSAGE):
                seen[nr, nc] = 1
                queue[tail, 0], queue[tail, 1] = nr, nc
                tail += 1
    return False


def is_solvable(maze: np.ndarray) -> bool:
    """Check four-neighbor connectivity between the size-relative entrance and exit."""
    if maze.ndim != 2 or maze.shape[0] != maze.shape[1]:
        raise ValueError(f"Expected a square maze, got shape {maze.shape}.")
    start, goal = maze_endpoints(maze.shape[0])
    if maze[start] != PASSAGE or maze[goal] != PASSAGE:
        return False
    return bool(
        _is_solvable_numba(
            np.ascontiguousarray(maze, dtype=np.uint8),
            start[0], start[1], goal[0], goal[1],
        )
    )


def solvability_rate(mazes: Iterable[np.ndarray]) -> float:
    """Fraction of generated mazes that connect entrance and exit."""
    items = list(mazes)
    if not items:
        raise ValueError("mazes cannot be empty")
    return float(np.mean([is_solvable(maze) for maze in items]))

