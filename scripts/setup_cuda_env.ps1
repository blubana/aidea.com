param(
    [string]$VenvPath = ".venv",
    [string]$Python = "python",
    [string[]]$PythonArgs = @()
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".")) {
    throw "Run this script from the project root."
}

& $Python @PythonArgs -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& "$VenvPath\Scripts\python.exe" -m pip install -r "requirements-base.txt"
& "$VenvPath\Scripts\python.exe" -m pip install -r "requirements-cuda.txt"

& "$VenvPath\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
