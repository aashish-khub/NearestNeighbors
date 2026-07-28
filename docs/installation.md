# Installation

## Requirements

- **Python 3.10 or later.** Check with `python --version`.
- No compiler or GPU is required; everything is pure Python on top of NumPy.

## Install from PyPI

```bash
pip install nsquared
```

Verify the install:

```bash
python -c "import nsquared; print(nsquared.get_available_datasets())"
```

You should see the registered benchmark datasets:

```
['heartsteps', 'movielens', 'prompteval', 'prop99', 'synthetic_data']
```

## Install from source

Use this if you want the latest unreleased code or intend to contribute.

**macOS / Linux:**

```bash
git clone https://github.com/aashish-khub/NearestNeighbors.git
cd NearestNeighbors
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/aashish-khub/NearestNeighbors.git
cd NearestNeighbors
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

`-e` installs in editable mode, so edits to `src/` take effect without reinstalling. The
`[dev]` extra adds `pytest`, `ruff`, and `pre-commit`.

## Dependencies

Installed automatically with the package:

| Package | Used for |
| --- | --- |
| `numpy` | All array computation |
| `pandas` | Dataset loading and reshaping |
| `hyperopt` | Distance-threshold search in the `FitMethod` classes |
| `matplotlib`, `seaborn` | Plotting utilities in `nsquared.utils.plotting_utils` |
| `joblib` | Caching of processed datasets |
| `requests`, `datasets` | Downloading benchmark data |
| `fancyimpute` | The SoftImpute baseline |
| `SyntheticControlMethods` | The synthetic control baseline for Prop 99 |
| `tabulate`, `tqdm` | Console output in the benchmark scripts |
| `wrds` | Optional access to WRDS for the earnings example |

The `dev` extra adds `pre-commit`, `ruff`, and `pytest`.

## Troubleshooting

**`python --version` reports 3.9 or earlier.** Install a newer Python. With
[pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation):

```bash
eval "$(pyenv init -)"
pyenv install 3.11
pyenv local 3.11
```

**VSCode is not picking up the virtual environment.** Run `Cmd/Ctrl + Shift + P` →
`Python: Select Interpreter` and choose the `.venv` interpreter.

**A dataset is missing from `get_available_datasets()`.** Loaders are discovered by
importing each subpackage of `nsquared.datasets`; a loader whose optional dependency is
not installed is skipped silently. Import it directly to see the underlying error:

```python
import importlib
importlib.import_module("nsquared.datasets.movielens")
```

**Benchmark scripts fail on Windows.** The experiment drivers in `bench/` are shell
scripts. Run them from [Git Bash](https://gitforwindows.org/); see
[`bench/README.md`](../bench/README.md).
