# setup_windows.ps1
# Windows setup script for VAE Project.
# Creates a virtual environment and installs dependencies.
# Run from the project root directory.

Write-Host "=== VAE Project: Windows Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Install Python 3.11+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-Not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "Virtual environment created at: .venv" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists at: .venv" -ForegroundColor Green
}

# Activate instructions
Write-Host ""
Write-Host "To activate the virtual environment, run:" -ForegroundColor Yellow
Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""

# Check if already activated
if ($env:VIRTUAL_ENV) {
    Write-Host "Virtual environment is already active." -ForegroundColor Green
} else {
    Write-Host "NOTE: Activate the virtual environment before running pip commands." -ForegroundColor Yellow
    Write-Host "  If you see an execution policy error, run:" -ForegroundColor Yellow
    Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
    Write-Host "  (Only if you understand and accept the security implications.)" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "After activating, run the following commands:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # Upgrade pip" -ForegroundColor DarkGray
Write-Host "  python -m pip install --upgrade pip" -ForegroundColor White
Write-Host ""
Write-Host "  # Install the project in editable mode with dev dependencies" -ForegroundColor DarkGray
Write-Host "  pip install -e '.[dev]'" -ForegroundColor White
Write-Host ""
Write-Host "  # Install PyTorch for CPU (default)" -ForegroundColor DarkGray
Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor White
Write-Host ""
Write-Host "  # For CUDA (GPU) support, visit https://pytorch.org to get the right command" -ForegroundColor DarkGray
Write-Host "  # for your CUDA version. Example for CUDA 12.1:" -ForegroundColor DarkGray
Write-Host "  # pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  # Verify installation" -ForegroundColor DarkGray
Write-Host "  python -c 'import vae_project; print(vae_project.__version__)'" -ForegroundColor White
Write-Host ""
Write-Host "=== Setup instructions complete ===" -ForegroundColor Cyan
