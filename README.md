# Discrete diffusion lab

Small research project for continuous Gaussian DDPM experiments of synthetic binary 32×32 mazes. The experiment notebook and reusable code live under `notebooks/`; architecture details are documented in `docs/`.

## Setup

```powershell
uv sync
uv run jupyter lab
```

The project installs CUDA 12.8-enabled PyTorch wheels for NVIDIA GPUs. Select the project `.venv` kernel in Jupyter, then open `notebooks/01_maze_discrete_diffusion.ipynb`. Run its cells manually. Training checkpoints are written to the ignored `artifacts/` directory.

