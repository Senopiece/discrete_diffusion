# Queens 8x8 · Discrete Flow Matching

One model transforms a full region coloring into queen colors and EMPTY cells.
The coloring is the initial state. Each model call receives the current grid,
optionally conditioned on time.

## Run

```powershell
uv sync
uv run python scripts/generate_queens8.py --count 10000000
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

The generator uses all logical CPU cores by default. GIL-free Numba workers
generate independent batches. Boundary moves update exact region-count
histograms only for placements using the changed cell; connectivity is checked
with 64-bit flood fills. Every accepted puzzle also passes an independent full
uniqueness check. `--workers 8` selects a worker count, and `--batch-size 256`
controls the number of candidates per batch. Output is reproducible for a fixed
seed and batch size, independently of worker count.

Generation writes preallocated arrays incrementally and saves progress every
five seconds in `data/queens8/.generation/`. Continue an interrupted run with
the same count, seed and batch size:

```powershell
uv run python scripts/generate_queens8.py --count 10000000 --resume
```

`--output PATH` selects a separate dataset directory. Completed arrays are
published through an atomic manifest update; partial generation leaves the
current dataset available. A compact fingerprint hash table performs exact
deduplication with bounded memory (about 554 MB for ten million puzzles).

The manifest references three NumPy arrays, all with dtype `uint8`:

- `regions.npy`, shape `(count, 8, 8)`: region labels.
- `queen_columns.npy`, shape `(count, 8)`: the queen's column in each row.
- `fingerprints.npy`, shape `(count, 32)`: SHA-256 symmetry fingerprints.

Together they use 104 bytes per puzzle: approximately 1.04 GB for ten million
puzzles, including fingerprints. `load_queens_dataset` opens read-only memory
maps. Training fetches individual pairs and constructs queen masks on demand;
readable puzzle IDs are derived only for selected indices. Integrity checks
stream file hashes and validate answers in bounded batches. Pass `verify=False`
to skip those scans when reopening an already verified dataset. Seeded splits
use indices and require about 80 MB for ten million puzzles. Checkpoints record
the ordered dataset's identity; array storage has separate file checksums.

Rules: one queen per row, column and region; queens cannot touch diagonally.
Longer diagonals are allowed. Training augments puzzles and solutions together
using rotations, reflections and color permutations.

## Training and evaluation

The endpoint model is a U-Net with 9 current-state channels and 2 fixed coordinate
channels. Its output is one erasure logit per cell,
shaped `(B,1,8,8)`. Sigmoid gives the probability of EMPTY in the endpoint.
Training uses binary cross-entropy from logits: target 1 for endpoint EMPTY
and 0 for a queen. Already-empty cells contribute zero, with averaging over
the full grid and batch. The U-Net uses base width 64 and time
width 128. Training batch size, augmented views per epoch and total epoch budget
are configured in the notebook.

An epoch draws `SAMPLES_PER_EPOCH` augmented pairs with replacement (4,096 by
default), giving 32 optimizer updates at batch size 128. The training plot
updates in place once per completed epoch and shows the sample-weighted mean
train BCE alongside validation BCE, with epoch number on the horizontal axis.
Validation uses a fixed subset of up to `VALIDATION_PUZZLES = 1024` puzzles at
six times; all sampling notebooks use the same fixed `TEST_PUZZLES = 256` test
puzzles. The subsets are prefixes of the seeded random held-out splits, and
their size limits are saved in the checkpoint. Full split membership remains
unchanged. Changing the validation subset clears its plotted validation history.

Checkpoints automatically save and resume
weights, optimizer, history, dataset hash, splits and RNG state to
`artifacts/queens8/queens8_dfm.pt`. With `RESUME = True`, increasing `EPOCHS`
continues from the saved optimizer step and history up to the new total budget.

`USE_TIME_CONDITIONING` controls the model's time branch. With `False`, the network
contains no timestep MLP or time-projection parameters and performs no time
embedding computations during training or inference. Time is still sampled to
construct training states and controls the flow sampler's transition rates.
The flag is saved with the architecture; sampling notebooks load it automatically.
Set `RESUME = False` to train a different architecture at the checkpoint path,
then use `RESUME = True` for continuation. Generation-step counts belong to
sampling configuration.

Trajectory examples use 100 stochastic steps with absorbing EMPTY cells.
Evaluation uses 1, 2, 4, 8, 16, 32 and 64 generation
steps on the fixed test subset (256 by default), with 32 real attempts
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

## Adaptive sampling with multiplicative risk

`notebooks/qeens/04_queens_product_risk_sampling.ipynb` orders cells without
replacement using their endpoint EMPTY probabilities. It erases the longest
prefix satisfying `1 - prod(p(EMPTY)) <= delta`, capped to leave eight cells.
The product is evaluated in log space. It estimates local risk under a
conditional-independence approximation.

The notebook shows trajectories and compares seven risk budgets using the saved
model and held-out split. Time follows completed deletions, and a zero-length
prefix forces one deletion. Success across attempts, model-call counts and
forced-step counts are saved to
`artifacts/queens8/queens8_dfm_product_risk_evaluation.npz`.
