# Discrete diffusion lab

Small research project comparing Gaussian DDPM, continuous flow matching, categorical D3PM, masked diffusion (MDLM), and categorical discrete flow matching on synthetic binary square mazes. Separate conditional DFM and binary D3PM experiments solve 7x7 colored-region Queens puzzles. The maze flow-matching notebook trains on 32x32 by default and evaluates transfer to 64x64 grids.

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

Shared Python code remains in `notebooks/_shared/`. Notebook setup finds the project root from the current directory or its ancestors, so imports and data/artifact paths also work when Jupyter starts inside an experiment folder. Run notebook cells manually. Training checkpoints are written to the ignored root `artifacts/` directory.

## Queens experiment

The Queens notebook reads the prepared collection in `data/queens7/`: 785 ready-made pairs from three external sources plus 5,000 synthetic puzzles grown round robin, with 2–12 cells per region and a unique solution verified offline. Notebook execution does not generate puzzles or search for their solutions. See `data/queens7/README.md` for provenance and preparation commands.

Both Queens notebooks include dataset galleries, coloring multiplicity statistics, forward trajectories, an updating logarithmic loss plot, checkpoint save/load cells, stochastic solution trajectories, strict held-out solution rates, retry curves, and a solvability study.

DFM transforms the coloring into colored queens plus EMPTY cells. D3PM instead generates a binary occupancy map from independent fair-bit noise. The unchanged seven-channel region coloring conditions its U-Net at every step, alongside the two-channel binary state and row/column coordinates. Training uses the D3PM variational objective with auxiliary clean-mask cross-entropy; reverse sampling uses categorical transitions without constraint repair.

The default budgets match: 100 epochs, 4,096 augmented examples per epoch, batch size 64 (6,400 optimizer steps), 100 network calls per generated attempt, and 32 attempts on each of the same 256 held-out puzzles. Both use the same split, geometric/color augmentations, U-Net widths, optimizer, and test-puzzle selection seed. Input/output heads differ, so total parameter counts and FLOPs are not exactly equal.

Checkpoints and evaluation caches have separate DFM/D3PM filenames in `artifacts/queens7/`. Load a checkpoint after the setup cells to skip training or resume it.

