# Installation

## Requirements

- **Python 3.10 or later.** Check with `python --version`.
- No compiler or GPU is required; everything is pure Python on top of NumPy.

## Install from PyPI

```bash
pip install nsquared
```

This installs the full estimator library — every nearest neighbor variant, every data
type, and cross-validation — and pulls in only `numpy`, `hyperopt`, and `tqdm`.

Verify it:

```bash
python -c "import nsquared; print(nsquared.get_available_datasets())"
```

```
['synthetic_data']
```

Only the synthetic generator is listed, because the real benchmark loaders need the
`data` extra (below).

## Optional extras

The heavier dependencies are opt-in, so a user who only wants to impute a matrix does not
have to install a data-download stack and a convex solver.

| Extra | Command | Adds | Pulls in |
| --- | --- | --- | --- |
| *(none)* | `pip install nsquared` | Imputers, data types, cross-validation, synthetic data | `numpy`, `hyperopt`, `tqdm` |
| `data` | `pip install "nsquared[data]"` | The $N^2$-Bench loaders | `pandas`, `joblib`, `requests`, `datasets` |
| `baselines` | `pip install "nsquared[baselines]"` | The SoftImpute baseline | `fancyimpute` |
| `plots` | `pip install "nsquared[plots]"` | `nsquared.utils.plotting_utils` | `matplotlib` |
| `examples` | `pip install "nsquared[examples]"` | Everything `examples/` and `bench/` import | the above, plus `seaborn`, `tabulate`, `SyntheticControlMethods`, `wrds` |
| `all` | `pip install "nsquared[all]"` | All of the above | — |
| `dev` | `pip install -e ".[dev]"` | `all` plus the test and lint toolchain | `pytest`, `ruff`, `pre-commit` |

Extras compose: `pip install "nsquared[data,baselines]"`.

With the `data` extra you should see all five loaders:

```bash
pip install "nsquared[data]"
python -c "import nsquared; print(nsquared.get_available_datasets())"
```

```
['heartsteps', 'movielens', 'prompteval', 'prop99', 'synthetic_data']
```

Nothing fails silently: asking for a loader whose extra is missing raises a `ValueError`
naming the extra to install, and `NNData.help()` lists unavailable datasets separately.

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
`[dev]` extra installs `[all]` plus `pytest`, `ruff`, and `pre-commit`, which is what the
test suite and CI expect.

## What each dependency is for

**Required:**

| Package | Used for |
| --- | --- |
| `numpy` | All array computation |
| `hyperopt` | Distance-threshold search in the `FitMethod` classes |
| `tqdm` | Progress reporting in `nsquared.estimation_methods` |

**Optional, by extra:**

| Package | Extra | Used for |
| --- | --- | --- |
| `pandas` | `data` | Dataset loading and reshaping |
| `joblib` | `data` | Caching of processed datasets |
| `requests` | `data` | Downloading MovieLens and Prop 99 |
| `datasets` | `data` | Downloading PromptEval from the Hugging Face Hub |
| `fancyimpute` | `baselines` | The SoftImpute baseline |
| `matplotlib` | `plots` | `nsquared.utils.plotting_utils` |
| `seaborn` | `examples` | Plots in `examples/simulations/` |
| `tabulate` | `examples` | Console tables in `examples/prompteval/` |
| `SyntheticControlMethods` | `examples` | The synthetic control comparison in `examples/prop99/` |
| `wrds` | `examples` | WRDS access for the earnings example |

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

**A dataset is missing from `get_available_datasets()`.** Almost always the `data` extra
is not installed: `pip install "nsquared[data]"`. `NNData.help()` lists which loaders are
unavailable and why, and `NNData.create()` on one of them raises a `ValueError` naming the
missing dependency. To see the raw import error:

```python
import importlib
importlib.import_module("nsquared.datasets.movielens")
```

**`ImportError` mentioning `fancyimpute` when using `softimpute`.** Install the baselines
extra: `pip install "nsquared[baselines]"`. `nsquared.baselines.usvt` needs only NumPy and
always works.

**`ModuleNotFoundError: No module named 'baselines'`.** As of the move into the package
namespace, the baselines live at `nsquared.baselines`. Change
`from baselines import usvt, softimpute` to
`from nsquared.baselines import usvt, softimpute`.

**Benchmark scripts fail on Windows.** The experiment drivers in `bench/` are shell
scripts. Run them from [Git Bash](https://gitforwindows.org/); see
[`bench/README.md`](../bench/README.md).
