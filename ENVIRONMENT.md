# Environment setup

This project currently keeps the EDA/prefix/feature pipeline dependency-free so
it can run before installing modeling libraries.

Detected GPU:

- NVIDIA GeForce RTX 4070 Laptop GPU
- Driver reports CUDA support: 13.2
- VRAM: 8 GB

## Recommended: uv environment

`uv` is available on this machine and is now the recommended environment manager.

Create a Python 3.12 CUDA-ready environment:

```powershell
.\scripts\setup_uv_env.ps1
```

Equivalent manual commands:

```powershell
uv python install 3.12
uv venv --python 3.12
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run current data pipeline with uv:

```powershell
uv run python src\eda_dataset.py
uv run python src\build_prefix_dataset.py --max-prefix-len 12
```

## Alternative: plain venv + pip

If you do not want to use uv, use the legacy setup script:

```powershell
.\scripts\setup_cuda_env.ps1
```

Then activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

PyTorch CUDA note:

- Use the stable PyTorch `cu130` wheels.
- A CUDA 13.2-capable driver can run CUDA 13.0 or CUDA 12.x bundled wheels.
- If Python 3.14 binary wheel compatibility becomes a blocker for any package,
  create the virtual environment with Python 3.12 or 3.13 instead:

```powershell
.\scripts\setup_cuda_env.ps1 -Python "py" -PythonArgs "-3.12"
```

## Planned acceleration usage

- LSTM/GRU models: PyTorch CUDA.
- CatBoost tabular baselines: `task_type="GPU"` if stable; fallback to CPU.
- XGBoost tabular experiments: `tree_method="hist", device="cuda"` if stable; fallback to CPU.
