# Discrete diffusion lab

Small research project comparing Gaussian DDPM, continuous flow matching, categorical D3PM, and masked diffusion (MDLM) on synthetic binary 32x32 mazes. The notebooks and reusable code live under `notebooks/`.

## Setup

```powershell
uv sync
uv run jupyter lab
```

The project installs CUDA 12.8-enabled PyTorch wheels for NVIDIA GPUs. Select the project `.venv` kernel in Jupyter, then open one of the experiment notebooks:

- `notebooks/01_maze_discrete_diffusion.ipynb` - Gaussian DDPM
- `notebooks/02_maze_flow_matching.ipynb` - continuous flow matching
- `notebooks/03_maze_d3pm.ipynb` - categorical D3PM
- `notebooks/04_maze_mdlm.ipynb` - MDLM-style absorbing masked diffusion

Run notebook cells manually. Training checkpoints are written to the ignored `artifacts/` directory.

