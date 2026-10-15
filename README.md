# Discrete diffusion lab

Small research project comparing Gaussian DDPM, continuous flow matching, categorical D3PM, masked diffusion (MDLM), and categorical discrete flow matching on synthetic binary square mazes. A separate conditional discrete flow-matching experiment solves 7x7 colored-region Queens puzzles. The maze flow-matching notebook trains on 32x32 by default and evaluates transfer to 64x64 grids. The notebooks and reusable code live under `notebooks/`.

## Setup

```powershell
uv sync
uv run jupyter lab
```

The project installs CUDA 12.8-enabled PyTorch wheels for NVIDIA GPUs. Select the project `.venv` kernel in Jupyter, then open one of the experiment notebooks:

- `notebooks/01_maze_ddpmh.ipynb` - Gaussian DDPM with embedding-delta prediction
- `notebooks/02_maze_fmh.ipynb` - continuous flow matching
- `notebooks/02_02_maze_fmhh.ipynb` - random-interval flow matching with learned-temperature KL
- `notebooks/03_maze_d3pm.ipynb` - categorical D3PM
- `notebooks/04_maze_mdlm.ipynb` - MDLM-style absorbing masked diffusion
- `notebooks/05_maze_discrete_flow_matching.ipynb` - categorical discrete flow matching with a uniform prior and learned endpoint posterior
- `notebooks/06_queens_discrete_flow_matching.ipynb` - conditional discrete flow matching for 7x7 colored-region Queens puzzles

Run notebook cells manually. Training checkpoints are written to the ignored `artifacts/` directory.

## Queens experiment

The Queens notebook reads the prepared collection in `data/queens7/`: 785 ready-made pairs from three external sources plus 5,000 synthetic puzzles grown round robin, with 2–12 cells per region and a unique solution verified offline. Notebook execution does not generate puzzles or search for their solutions. See `data/queens7/README.md` for provenance and preparation commands.

It includes categorical forward trajectories, an updating logarithmic cross-entropy plot, checkpoint save/load cells, stochastic solution trajectories, and strict held-out solution rates. Sampling uses 100 DFM steps by default.

