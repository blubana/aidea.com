param(
    [string]$Python = "3.12"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath "pyproject.toml")) {
    throw "Run this script from the project root."
}

uv python install $Python
uv venv --python $Python
uv sync
uv run python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
