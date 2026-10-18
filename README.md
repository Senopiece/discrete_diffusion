# Discrete diffusion lab

Small research project comparing Gaussian DDPM, continuous flow matching, categorical D3PM, masked diffusion (MDLM), and categorical discrete flow matching on synthetic binary square mazes. Separate conditional DFM, binary D3PM, and continuous flow-matching experiments solve 7x7 colored-region Queens puzzles. The maze flow-matching notebook trains on 32x32 by default and evaluates transfer to 64x64 grids.

## Setup

```powershell
uv sync
uv run jupyter lab
```

The project installs CUDA 12.8-enabled PyTorch wheels for NVIDIA GPUs. Select the project `.venv` kernel in Jupyter, then open one of the experiment notebooks:

- `notebooks/mazes/01_maze_ddpmh.ipynb` - Gaussian DDPM with embedding-delta prediction
- `notebooks/mazes/02_maze_fmh.ipynb` - continuous flow matching
- `notebooks/mazes/03_maze_d3pm.ipynb` - categorical D3PM
- `notebooks/mazes/04_maze_mdlm.ipynb` - MDLM-style absorbing masked diffusion
- `notebooks/mazes/05_maze_discrete_flow_matching.ipynb` - categorical discrete flow matching with a uniform prior and learned endpoint posterior
- `notebooks/qeens/01_queens_discrete_flow_matching.ipynb` - conditional discrete flow matching for Queens
- `notebooks/qeens/02_queens_d3pm.ipynb` - binary D3PM conditioned on the fixed region coloring
- `notebooks/qeens/03_queens_cfm.ipynb` - coloring-to-solution continuous flow matching with endpoint-residual MSE and deterministic ODE generation

Shared Python code remains in `notebooks/_shared/`. Notebook setup finds the project root from the current directory or its ancestors, so imports and data/artifact paths also work when Jupyter starts inside an experiment folder. Run notebook cells manually. Training checkpoints are written to the ignored root `artifacts/` directory.

## Queens experiment

The Queens notebook reads the prepared collection in `data/queens7/`: 785 ready-made pairs from three external sources plus 5,000 synthetic puzzles grown round robin, with 2–12 cells per region and a unique solution verified offline. Notebook execution does not generate puzzles or search for their solutions. See `data/queens7/README.md` for provenance and preparation commands.

All Queens notebooks include dataset galleries, coloring multiplicity statistics, forward trajectories, an updating logarithmic loss plot, checkpoint save/load cells, generation trajectories, and strict held-out solution rates. DFM and D3PM also include retry curves and a solvability study.

The DFM notebook also compares each evaluated test puzzle's measured success probability with the ordered human-style deduction sequence from `gitars/queens-generator`, pinned in `uv.lock`. This sequence includes one-level lookahead, but does not recursively search for a complete solution. A 100% stacked rectangle shows the fraction solved deductively (green, above) and not solved (red, below) at each measured DFM probability. Empty probability groups are gray. Verdicts and ordered deduction traces are cached by puzzle fingerprint, with per-puzzle JSONL/CSV and comparison arrays saved under `artifacts/queens7/`; the section can reuse the saved DFM evaluation without running the network again. Section 8.1 derives a trace metric (mean explicit elimination deductions before each queen placement) and plots it against per-puzzle DFM solvability, with source colors, grouped means and bootstrap intervals. This is exploratory analysis of the measured test results, with outputs saved as `queens7_deduction_wait_metric.*`.

Section 8.2 reconstructs queen-to-queen automatic-exclusion supports and measures their maximum and mean dependency depth, using `notebooks/_shared/queens_deduction_metrics.py`. It compares the new metric with the waiting metric on the same puzzles and draws example dependency graphs. Explicit elimination deductions are treated as supplied lemmas; unexpanded advanced-rule placements are marked as opaque seeds. This measures queen-cascade depth rather than complete proof depth. Results and graphs are saved as `queens7_dependency_depth.*` and `queens7_dependency_graphs.*`.

Sections 8.3 and 8.4 extract 51 deduction-trace features and estimate their information about measured DFM solvability. Individual-feature information uses discretization and a shuffled-label reference; whole-vector information uses an out-of-fold classifier to estimate a variational lower bound for four solvability ranges. Decision trees, random forests, and gradient boosting predict per-puzzle success rates with nested cross-validation grouped by queen placement, including rotations and reflections. A source-only baseline and conditional information audit check whether predictions mainly reflect dataset provenance. This remains exploratory analysis of the finite-attempt evaluation, not a precise estimate of total mutual information. Feature extraction and fitting live in `notebooks/_shared/queens_solvability_prediction.py`; features, predictions, plots, fitted models, and cache metadata are saved as `queens7_deduction_features.csv`, `queens7_feature_information.*`, and `queens7_solvability_*` under `artifacts/queens7/`.

DFM transforms the coloring into colored queens plus EMPTY cells. D3PM instead generates a binary occupancy map from independent fair-bit noise. The unchanged seven-channel region coloring conditions its U-Net at every step, alongside the two-channel binary state and row/column coordinates. Training uses the D3PM variational objective with auxiliary clean-mask cross-entropy; reverse sampling uses categorical transitions without constraint repair.

Training defaults match across the three experiments: 100 epochs, 4,096 augmented examples per epoch, batch size 64 (6,400 optimizer steps), and 100 network calls per generated answer. They share the split, geometric/color augmentations, U-Net widths, optimizer, and test-puzzle selection seed. Input/output heads differ: CFM uses 10 input channels and 8 output channels, DFM uses 17 and 8, and D3PM uses 11 and 2. Parameter counts and FLOPs therefore differ slightly.

CFM follows the linear path `xt = (1-t) * x0 + t * x1`, where `x0` is the complete one-hot coloring and `x1` retains region colors at queen cells and uses EMPTY elsewhere. Its U-Net receives only the evolving eight-channel state, row/column coordinates, and time; the original coloring is not provided again as a condition. It learns velocity using only `MSE((1-t) * v(xt,t), x1 - xt)`. Generation uses deterministic Euler ODE integration from the coloring, then argmax decoding over all eight classes without a region-color mask. Trajectory colors are mixed directly from embedding components, so the forward display fades linearly into EMPTY. Learned states outside the probability simplex are projected onto it for display only; no visualization softmax sharpens the transition.

DFM/D3PM evaluation uses 32 attempts per puzzle. CFM evaluates the same 256 held-out puzzles with one deterministic ODE answer per puzzle and puzzle-bootstrap confidence intervals. Cached example boards show the initial coloring, generated colored-queens/EMPTY answer, and supplied target. Trajectory examples use separate validation puzzles.

Checkpoints and evaluation caches have separate filenames in `artifacts/queens7/`. CFM uses `queens7_cfm_coloring_endpoint_residual.pt` and `queens7_cfm_coloring_endpoint_residual_evaluation.npz`. Its new source path and input channels require fresh training; earlier Gaussian-source CFM checkpoints are incompatible. Load a matching checkpoint after the setup cells to skip training or resume it.

