# Queens 8x8 · Discrete Flow Matching

One model transforms a full region coloring into queen colors and EMPTY cells.
The coloring is the initial state. Each model call receives the current grid
and time.

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

`data/queens8/` stores synthetic puzzle/solution pairs. Each puzzle has
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
the categorical coloring-to-solution path. The U-Net uses base width 64 and time
width 128. Training batch size, augmented views per epoch and total epoch budget
are configured in the notebook.

The training plot updates in place. Checkpoints automatically save and resume
weights, optimizer, history, dataset hash, splits and RNG state under
`artifacts/queens8/`. Increase `EPOCHS` to extend the total budget.

Trajectory examples use 100 stochastic steps with absorbing EMPTY cells.
Evaluation uses 1, 2, 4, 8, 16, 32 and 64 generation
steps on the same selected test puzzles (256 by default), with 32 real attempts
per puzzle for each step budget. One plot compares all seven empirical pass@k
curves in different colors, averaged over subsets of the recorded attempts, with
95% bands from bootstrapping whole puzzles. Results include the step budgets,
checkpoint and dataset hashes. Running evaluation generates fresh attempts.

## Confidence-guided sampling

After training in notebook 01, open
`notebooks/qeens/02_queens_confidence_sampling.ipynb`. It loads the saved model
and its held-out split. Cells are erased without replacement with
weights `p(EMPTY)**(1/tau)`, using cumulative quotas that leave exactly eight
colored cells. Evaluation checks the remaining Queens constraints.

The notebook shows trajectories and evaluates temperatures 0.25, 1 and 4 across
1, 2, 4, 8, 16, 32 and 64 model calls. Every combination uses the same 256 test
puzzles and 32 attempts per puzzle. The plot has independent temperature and
step selectors: color identifies temperature, line style identifies steps,
and optional bands show 95% puzzle-bootstrap intervals.
Results are saved to `artifacts/queens8/queens8_dfm_confidence_evaluation.npz`.

## Adaptive risk-budget sampling

`notebooks/qeens/03_queens_adaptive_sampling.ipynb` loads the trained model and
chooses the number of erasures at each iteration. Cells are ordered with weights
`p(EMPTY)**(1/tau)`. The sampler erases the longest prefix whose cumulative
`sum(1-p(EMPTY))` fits the risk budget, capped to leave at least eight cells.
An empty prefix forces one deletion. Time follows the fraction of the 56
required deletions already completed. Generation finishes with exactly eight
cells in 1–56 model calls.

The notebook shows trajectories, compares success across attempts for three
temperatures and seven risk budgets, and plots the observed sampling cost.
Per-attempt outcomes, model-call counts and forced-step counts are saved to
`artifacts/queens8/queens8_dfm_adaptive_evaluation.npz`. The risk budget is a local
estimate based on the model probabilities; forced steps can exceed it.
