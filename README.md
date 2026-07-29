# $N^2$: Nearest Neighbors for Matrix Completion

[![CI](https://github.com/aashish-khub/NearestNeighbors/actions/workflows/ci.yml/badge.svg)](https://github.com/aashish-khub/NearestNeighbors/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nsquared.svg)](https://pypi.org/project/nsquared/)
[![Python](https://img.shields.io/pypi/pyversions/nsquared.svg)](https://pypi.org/project/nsquared/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.04166-b31b1b.svg)](https://arxiv.org/abs/2506.04166)

$N^2$ is a Python package that unifies the family of **nearest neighbor (NN) methods for
matrix completion** behind a single interface, together with **$N^2$-Bench**, a benchmark
suite of real-world matrix completion tasks.

Given a partially observed matrix — users × movies, patients × time points, models ×
prompts — the goal is to fill in the missing entries. $N^2$ implements the modern NN
estimators for this problem, works for entries that are *scalars* **or** entire
*distributions*, and lets you switch between methods by changing one line.

---

## Why $N^2$?

Nearest neighbor methods impute one entry at a time by matching on similar rows or
columns. Unlike spectral and nuclear-norm methods, they remain reliable when missingness
is driven by the entries themselves or by unobserved confounders — the common case in
health, economics, and recommendation data.

Recent years have produced a rapid succession of NN variants (two-sided, doubly robust,
adaptively weighted, distributional), each released as a standalone research
implementation with its own data conventions and hyperparameter interface. That makes
them hard to compare and hard to adopt. $N^2$ solves this:

- **One interface for every variant.** Compare row-wise, column-wise, two-sided, doubly
  robust, adaptively-weighted, and distributional NN by changing one argument.
- **Scalars and distributions.** Entries can be single numbers or empirical
  distributions, with the same estimator code path for both.
- **Extensible by design.** Add a new estimator and it works for every data type; add a
  new data type and it works with every estimator.
- **A real benchmark.** $N^2$-Bench auto-downloads four real-world datasets across mobile
  health, recommender systems, causal panel data, and LLM evaluation, so new methods can
  be stress-tested beyond synthetic low-rank matrices.
- **Baselines included.** USVT and SoftImpute are packaged alongside for comparison.

## Implemented methods

| Method | Constructor / class | Scalar | Distributional | Reference |
| --- | --- | :---: | :---: | --- |
| Row-wise NN | `row_row()` | ✅ | ✅ | [Li et al., 2019](https://arxiv.org/abs/1705.04867), [Dwivedi et al., 2022](https://arxiv.org/abs/2202.06891) |
| Column-wise NN | `col_col()` | ✅ | ✅ | as above |
| Two-sided NN | `ts_nn()` | ✅ | ✅ | [Sadhukhan et al., 2024](https://arxiv.org/abs/2411.12965) |
| Doubly robust NN | `dr_nn()` | ✅ | — | [Dwivedi et al., 2022](https://arxiv.org/abs/2211.14297) |
| Adaptively-weighted NN | `aw_nn()` | ✅ | ✅ | [Sadhukhan et al., 2025](https://arxiv.org/abs/2505.09612) |
| Auto NN | `AutoEstimator` | ✅ | — | [Chin et al., 2025](https://arxiv.org/abs/2506.04166) |
| Nadaraya–Watson (kernel-smoothed) NN | `NadarayaWatsonEstimator` | ✅ | — | [Nadaraya, 1964](https://doi.org/10.1137/1109020); Watson, 1964 |
| Kernel (MMD) distributional NN | `DistributionKernelMMD` | — | ✅ | [Choi et al., 2024](https://arxiv.org/abs/2410.13381) |
| Wasserstein distributional NN | `DistributionWasserstein*` | — | ✅ | [Feitelberg et al., 2024](https://arxiv.org/abs/2410.13112) |
| USVT (baseline) | `nsquared.baselines.usvt` | ✅ | — | [Chatterjee, 2015](https://doi.org/10.1214/14-AOS1272) |
| SoftImpute (baseline) | `nsquared.baselines.softimpute` | ✅ | — | [Hastie et al., 2015](https://jmlr.org/papers/v16/hastie15a.html) |

Doubly robust estimation requires a well-defined subtraction on entries, so it is
restricted to scalar (vector-space) entries.

---

## Installation

$N^2$ requires **Python 3.10 or later**. Check with `python --version`.

```bash
pip install nsquared
```

That gives you every imputer, every data type, and cross-validation, on a deliberately
small dependency footprint (`numpy`, `hyperopt`, `tqdm`). The heavier pieces are opt-in:

| Install | Adds |
| --- | --- |
| `pip install nsquared` | The imputers, data types, cross-validation, synthetic data, both baselines |
| `pip install "nsquared[data]"` | The $N^2$-Bench loaders — HeartSteps, MovieLens, PromptEval, Prop 99 |
| `pip install "nsquared[plots]"` | Plot styling helpers (`matplotlib`) |
| `pip install "nsquared[examples]"` | Everything the scripts in `examples/` and `bench/` need |
| `pip install "nsquared[all]"` | All of the above |

If you ask for a benchmark dataset without its extra, the error tells you which one to
install — nothing fails silently.

<details>
<summary>Installing from source (latest, unreleased code)</summary>

**macOS / Linux:**

```bash
git clone https://github.com/aashish-khub/NearestNeighbors.git
cd NearestNeighbors
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

**Windows:**

```powershell
git clone https://github.com/aashish-khub/NearestNeighbors.git
cd NearestNeighbors
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

</details>

<details>
<summary>Setting the Python version with pyenv</summary>

If you are not on Python 3.10+, [install pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation),
run `eval "$(pyenv init -)"`, then `pyenv local 3.11`.

</details>

**Dependencies.** The core install requires only `numpy`, `hyperopt` (threshold search),
and `tqdm` (progress reporting). Everything else is behind the extras in the table above.
The `[dev]` extra installs `[all]` plus `pytest`, `ruff`, and `pre-commit`.

> [!NOTE]
> If using VSCode, set the interpreter to the `.venv` environment with
> `Cmd/Ctrl + Shift + P` → `Python: Select Interpreter`.

---

## Quickstart

Impute a missing entry with row-wise nearest neighbors, tuning the distance threshold by
cross-validation:

```python
import numpy as np
from nsquared import NNData, row_row, Scalar, LeaveBlockOutValidation

# 1. Get a partially observed matrix and its missingness mask.
#    mask[i, t] == 1 means entry (i, t) was observed.
loader = NNData.create("synthetic_data", num_rows=50, num_cols=50, seed=0, miss_prob=0.3)
data, mask = loader.process_data_scalar()

# 2. Choose an estimator. Swap row_row() for ts_nn(), dr_nn(), aw_nn(), ...
imputer = row_row()

# 3. Tune the distance threshold on a held-out block of observed entries.
rng = np.random.default_rng(0)
observed = np.argwhere(mask == 1)
block = [tuple(map(int, observed[i])) for i in rng.choice(len(observed), 30, replace=False)]

cv = LeaveBlockOutValidation(
    block=block,
    distance_threshold_range=(0, 1),
    n_trials=20,
    data_type=Scalar(),
)
cv.fit(data, mask, imputer)

# 4. Impute.
print(imputer.impute(row=0, column=0, data_array=data, mask_array=mask))
```

Switching methods is a one-line change:

```python
from nsquared import ts_nn, dr_nn, aw_nn

imputer = ts_nn(distance_threshold_row=0.5, distance_threshold_col=0.5)
value = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
```

See [`docs/quickstart.md`](docs/quickstart.md) for distributional matrix completion,
imputing a whole matrix, and using the real benchmark datasets.

---

## Documentation

| | |
| --- | --- |
| [Installation](docs/installation.md) | Requirements, install options, troubleshooting |
| [Quickstart](docs/quickstart.md) | Scalar and distributional worked examples |
| [User guide](docs/user_guide.md) | The `DataType` / `EstimationMethod` framework and how to extend it |
| [API reference](docs/api_reference.md) | Every public class and function |
| [Datasets](docs/datasets.md) | The $N^2$-Bench loaders and their options |
| [Examples](examples/) | Runnable scripts per dataset |
| [Benchmark](bench/README.md) | Reproducing the paper's experiments; adding methods and datasets |

## $N^2$-Bench

To reproduce the experiments in [our paper](https://arxiv.org/abs/2506.04166) and test new
methods or datasets, see the [`bench`](./bench/) directory. For direct access to the exact
matrices used in the benchmark, we host the data
[in a companion repository](https://github.com/calebchin/nsquared_bench_data).

---

## Contributing

We welcome contributions — bug reports, questions, documentation, new estimators, new data
types, and new benchmark datasets. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment, run the
checks, and open a pull request, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for our
community standards.

**Getting help:** open an issue using the
[Question / support template](https://github.com/aashish-khub/NearestNeighbors/issues/new/choose).
**Reporting a bug:** use the Bug report template and include a minimal reproduction.

### Development checks

```bash
pre-commit run --all-files   # ruff lint + format, pyright type checking
pytest                       # test suite
```

Both run in [CI](.github/workflows/ci.yml) on every push and pull request across Python
3.10–3.12. Release instructions are in [RELEASE.md](RELEASE.md).

---

## Citation

If you use $N^2$ in your research, please cite the accompanying paper:

```bibtex
@article{chin2025nsquared,
  title={N$^2$: A unified Python package and test bench for nearest neighbor-based matrix completion},
  author={Chin, Caleb and Khubchandani, Aashish and Maskara, Harshvardhan and Choi, Kyuseong and Feitelberg, Jacob and Gong, Albert and Paul, Manit and Sadhukhan, Tathagata and Agarwal, Anish and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2506.04166},
  year={2025}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

<details>
<summary>Papers whose methods are implemented here</summary>

```bibtex
@article{dwivedi2022counterfactual,
  title={Counterfactual inference for sequential experiments},
  author={Dwivedi, Raaz and Tian, Katherine and Tomkins, Sabina and Klasnja, Predrag and Murphy, Susan and Shah, Devavrat},
  journal={arXiv preprint arXiv:2202.06891},
  year={2022}
}

@article{sadhukhan2024adaptivity,
  title={On adaptivity and minimax optimality of two-sided nearest neighbors},
  author={Sadhukhan, Tathagata and Paul, Manit and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2411.12965},
  year={2024}
}

@article{dwivedi2022doubly,
  title={Doubly robust nearest neighbors in factor models},
  author={Dwivedi, Raaz and Tian, Katherine and Tomkins, Sabina and Klasnja, Predrag and Murphy, Susan and Shah, Devavrat},
  journal={arXiv preprint arXiv:2211.14297},
  year={2022}
}

@article{feitelberg2024distributional,
  title={Distributional matrix completion via nearest neighbors in the Wasserstein space},
  author={Feitelberg, Jacob and Choi, Kyuseong and Agarwal, Anish and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2410.13112},
  year={2024}
}

@article{choi2024learning,
  title={Learning counterfactual distributions via kernel nearest neighbors},
  author={Choi, Kyuseong and Feitelberg, Jacob and Chin, Caleb and Agarwal, Anish and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2410.13381},
  year={2024}
}

@article{sadhukhan2025adaptively,
  title={Adaptively-weighted nearest neighbors for matrix completion},
  author={Sadhukhan, Tathagata and Paul, Manit and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2505.09612},
  year={2025}
}
```

</details>

## License

MIT — see [LICENSE](LICENSE).
