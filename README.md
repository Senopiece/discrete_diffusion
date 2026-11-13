# Queens 8x8 · Discrete Flow Matching

One model transforms a full region coloring into queen colors and EMPTY cells.
The coloring enters through the initial state only; later steps receive the
current grid and time, with no separate coloring condition.

## Run

```powershell
uv sync
uv run python scripts/generate_queens8.py --count 5000
```

Generate the dataset once, then open
`notebooks/qeens/01_queens_discrete_flow_matching.ipynb` with the project `.venv`
kernel. The notebook only loads the prepared pairs and contains training, solution
trajectories, and probability of at least one solution versus number of attempts.

## Data

`data/queens8/` contains 5,000 synthetic puzzle/solution pairs. Each puzzle has
eight connected regions of 3–14 cells and exactly one solution. The offline
generator plants a legal placement, grows regions, refines their boundaries while
preserving connectivity, and checks uniqueness against every row/column
permutation without touching queens. Rotations, reflections and color relabelings
are deduplicated before the train/validation/test split.

Rules: one queen per row, column and region; queens cannot touch diagonally.
Longer diagonals are allowed. Training augments puzzles and solutions together
using rotations, reflections and color permutations.

## Training and evaluation

The endpoint model is a U-Net with 9 current-state channels and 2 fixed coordinate
channels, conditioned on scalar time. Train using mean pixel cross-entropy along
the categorical coloring-to-solution path. Defaults: base width 64, time width
128, batch size 128, 4,096 augmented views per epoch and 440 total epochs.

The training plot updates in place. Checkpoints automatically save and resume
weights, optimizer, history, dataset hash, splits and RNG state under
`artifacts/queens8/`. Increase `EPOCHS` to extend the total budget.

Sampling uses 100 stochastic steps with absorbing EMPTY cells, without search or
repair. Evaluation runs 32 real attempts per selected test puzzle (256 by default)
and plots empirical pass@k, averaged over subsets of the recorded attempts, with
a 95% interval from bootstrapping whole puzzles. Results include the checkpoint
and dataset hashes. Evaluation recomputes outcomes when its cell is run.
