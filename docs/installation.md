# Installation (Windows)

All commands are for **PowerShell** and are run from the repository root
(the folder that contains `pyproject.toml`).

## 1. Prerequisites

* Windows 10/11, 64-bit.
* **Python 3.11 or newer** (`requires-python = ">=3.11"` in
  [`pyproject.toml`](../pyproject.toml)). Install from https://www.python.org
  and tick "Add python.exe to PATH".
* PowerShell 5.1 or newer (built into Windows).
* Internet access for the first dataset download and for `pip`.
* Optional: NVIDIA GPU with a recent driver for CUDA. CPU is sufficient.

Check which Python is the default:

```powershell
python --version
py -0p        # lists every Python installed via the Windows launcher
```

If `python --version` reports a very new interpreter (for example 3.14), PyTorch
or other binary dependencies may not publish wheels for it yet. In that case
create the environment with an interpreter that PyTorch supports (check the
selector on https://pytorch.org), for example:

```powershell
py -3.12 -m venv .venv
```

## 2. Get the code

```powershell
git clone https://github.com/joaopaulo164/vae-project.git
cd vae-project
```

Replace the URL with the real repository address.

## 3. Create the virtual environment

```powershell
python -m venv .venv
```

or run the helper, which creates `.venv` and prints the remaining steps
(it does not install anything itself):

```powershell
.\scripts\setup_windows.ps1
```

## 4. Activate it

```powershell
.venv\Scripts\Activate.ps1
```

The prompt should show `(.venv)`. If PowerShell reports that running scripts
is disabled, see "Execution policy" in [troubleshooting.md](troubleshooting.md).
A policy-free alternative is `cmd`:

```powershell
cmd /c ".venv\Scripts\activate.bat && python --version"
```

## 5. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 6. Install PyTorch first (recommended)

Installing PyTorch **before** the project lets you choose the CPU or CUDA
build explicitly. If you install the project first, pip will pull the default
PyTorch wheel from PyPI.

### CPU build (default, works everywhere)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA build (NVIDIA GPU)

1. Check the driver: `nvidia-smi` (shows the highest CUDA version the driver
   supports).
2. Open https://pytorch.org, choose *Stable / Windows / Pip / Python /
   CUDA x.y* and copy the command it generates. Example for CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. You do not need a separate CUDA Toolkit install: the PyTorch wheels bundle
   the CUDA runtime. You do need a driver that is new enough.

If a CPU build was installed earlier, uninstall it first:

```powershell
pip uninstall -y torch torchvision
```

## 7. Install the project (editable)

```powershell
pip install -e ".[dev]"
```

`-e` (editable) installs the package from `src/` so code changes take effect
without reinstalling; `[dev]` adds pytest, ruff, black, mypy, bandit and
pre-commit. For runtime dependencies only: `pip install -e .`. As an
alternative `pip install -r requirements.txt` installs just the libraries
without registering the `vae_project` package, so `python -m vae_project.cli`
would not be importable from other directories.

Dependencies (from `pyproject.toml`): torch>=2.2.0, torchvision>=0.17.0,
numpy>=1.26.0, pandas>=2.2.0, scikit-learn>=1.4.0, umap-learn>=0.5.5,
matplotlib>=3.8.0, seaborn>=0.13.0, PyYAML>=6.0.1, tqdm>=4.66.0,
psutil>=5.9.0.

## 8. Verify the installation

```powershell
# Package importable and version
python -c "import vae_project; print(vae_project.__version__)"

# PyTorch build and device
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available(), torch.version.cuda)"

# Other key libraries
python -c "import torchvision, sklearn, umap, matplotlib, pandas; print('imports OK')"

# CLI and config validation (no download)
python -m vae_project.cli --help
python -m vae_project.cli validate-config --config configs/mnist_unsupervised.yaml

# End-to-end check with synthetic data (no download, no GPU)
python -m vae_project.cli smoke-test

# Unit and integration tests
python -m pytest
```

`torch.cuda.is_available()` printing `False` is normal for the CPU build.
With `runtime.device: auto` the project then uses the CPU.

## 9. Known packaging caveat

`pyproject.toml` declares `build-backend = "setuptools.backends.legacy:build"`.
That is not a standard setuptools backend path (the standard ones are
`setuptools.build_meta` and `setuptools.build_meta:__legacy__`). I could not
run `pip install -e` in the environment used to write this documentation, so
I have not confirmed whether this fails on your machine. If the editable
install stops with a backend import error, change that single line to
`build-backend = "setuptools.build_meta"` and retry.

## 10. Updating and removing

```powershell
git pull
pip install -e ".[dev]"      # only needed if dependencies changed
deactivate                   # leave the venv
Remove-Item -Recurse -Force .venv   # delete it
```
