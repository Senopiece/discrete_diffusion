# Queens discrete generation

Compare discrete flow matching (DFM) and binary categorical diffusion (D3PM) on 7x7 colored-region Queens puzzles.

## Setup

```powershell
uv sync
```

Select the project `.venv` Python kernel in your notebook editor and open:

- `notebooks/qeens/01_queens_discrete_flow_matching.ipynb` - coloring-to-solution DFM, including an initial-state-only conditioning ablation.
- `notebooks/qeens/02_queens_d3pm.ipynb` - binary D3PM conditioned on the original coloring, including comparison with a trained DFM checkpoint.

PyTorch uses the configured CUDA 12.8 wheel index. Notebook setup finds the project root from the current directory or its ancestors. Shared code lives in `notebooks/_shared/`; checkpoints and evaluation caches are saved under the ignored root `artifacts/queens7/` directory.

## Dataset and rules

The prepared dataset in `data/queens7/` contains 785 external puzzle/solution pairs and 5,000 synthetic pairs prepared offline. Each puzzle needs exactly one queen in every row, column, and region. Queens may not touch diagonally; longer diagonals are allowed.

The notebooks load stored pairs without generating puzzles or searching for solutions. Splitting precedes augmentation and groups duplicate colorings up to rotations, reflections, and color relabeling. Training applies paired rotations/reflections and color permutations; validation and test remain unchanged.

The dataset loading and collection code retains source provenance. To rebuild the collection outside the notebooks:

```powershell
uv run python scripts/collect_queens7.py
uv run python scripts/generate_queens7.py --count 5000
```

See `data/queens7/README.md` for source details and offline uniqueness verification.

## Models and training

DFM starts from the full coloring and generates the original color at queen cells and EMPTY elsewhere. Its U-Net receives the eight-state grid, the seven-channel original coloring, two coordinate channels, and time. Training uses endpoint cross-entropy; sampling follows the categorical flow without constraint repair. The initial-state-only variant instead receives just the eight-state grid, two coordinates, and time (10 input channels). It starts from the complete coloring and never receives the original map again. Its output support is derived solely from the current state, including absorbing EMPTY cells.

D3PM generates a binary queen occupancy map from independent fair-bit noise. The original coloring conditions every reverse step. Its U-Net uses two state channels, seven coloring channels, two coordinate channels, and time. Training combines the reverse variational objective with auxiliary clean-mask cross-entropy. Sampling uses the corresponding categorical reverse posterior without repairing the result.

Both models use the shared U-Net with base width 64 and time-embedding width 128. The notebook defaults use batch size 128 and 4,096 augmented examples per epoch: 440 epochs for DFM and 750 for D3PM. Checkpoints record actual optimizer updates, training history, dataset fingerprint, split, optimizer state, and RNG state. Sampling uses 100 steps. Matching sampling budgets does not imply equal training budgets.

## Evaluation and comparisons

The notebooks include dataset examples, coloring multiplicity statistics, forward paths, updating training plots, checkpoint save/load cells, and stochastic solution trajectories.

Evaluation uses the same held-out puzzle selection and 32 attempts per puzzle. It reports strict rule validity, exact target agreement, source-specific solve rates, and the probability of at least one solution versus the number of attempts. Confidence intervals bootstrap entire puzzles to preserve repeated-attempt groups.

DFM Section 5.1 trains the initial-state-only variant from scratch with the same U-Net widths, CE objective, split, augmentations, optimizer settings, and total update target as the saved coloring-conditioned reference. Its default seed matches the reference seed; the smaller input head changes the parameter count slightly. It saves under a distinct checkpoint format and filename, without loading the retired second-run weights. Section 6.1 shows its solution trajectories. Section 7.1 compares repeated coloring conditioning against initial-state-only generation using paired retry curves and a per-puzzle scatter with an attempt-count slider. Each slider position k shows the probability of at least one solution within k attempts, estimated from all subsets of the recorded attempts. New comparison caches use the `queens7_dfm_conditioning_*` prefix.

D3PM Section 7.1 loads the trained DFM independently of its own model and evaluates the same shared held-out puzzles with matching attempts and sampling steps. The comparison reports per-puzzle solvability and actual training counts.

Comparison caches are validated against checkpoint SHA-256, dataset fingerprint, puzzle IDs, and sampling settings. The primary checkpoint names are `queens7_discrete_flow_matching.pt`, `queens7_discrete_flow_matching_initial_only.pt`, and `queens7_d3pm.pt`.
