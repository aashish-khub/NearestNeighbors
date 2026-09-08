# $N^2$ documentation

$N^2$ is a Python package for **matrix completion with nearest neighbors**. It unifies the
modern family of nearest neighbor (NN) estimators behind a single interface, supports
entries that are scalars *or* probability distributions, and ships $N^2$-Bench, a
benchmark of real-world matrix completion tasks.

## The problem

You have an $N \times T$ matrix in which only some entries are observed:

| | movie 1 | movie 2 | movie 3 |
| --- | --- | --- | --- |
| **user 1** | 4 | ? | 2 |
| **user 2** | ? | 5 | ? |
| **user 3** | 3 | 5 | ? |

*Matrix completion* is the task of estimating the `?` entries. This same shape appears in
panel data (units × time), mobile health (patients × decision points), and LLM evaluation
(models × prompts).

Nearest neighbor methods estimate a single missing entry $(i, t)$ by finding rows similar
to row $i$ (or columns similar to column $t$) among the entries that *were* observed, then
averaging their values at that position. Because they operate one entry at a time and
match on observed data, they stay reliable when the missingness depends on the entries
themselves or on unobserved confounders — where spectral and nuclear-norm methods degrade.

## Where to go next

| Page | What it covers |
| --- | --- |
| [Installation](installation.md) | Requirements, install options, troubleshooting |
| [Quickstart](quickstart.md) | Worked scalar and distributional examples |
| [User guide](user_guide.md) | The two-abstraction design, and how to extend it |
| [API reference](api_reference.md) | Every public class and function |
| [Datasets](datasets.md) | The $N^2$-Bench loaders and their options |

Related material lives outside the `docs/` tree:

- [`examples/`](../examples/) — runnable scripts, one directory per dataset.
- [`bench/README.md`](../bench/README.md) — reproducing the paper's experiments end to end.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — development setup, checks, and PR process.

## A 30-second example

```python
from nsquared import NNData, row_row

loader = NNData.create("synthetic_data", num_rows=50, num_cols=50, seed=0)
data, mask = loader.process_data_scalar()

imputer = row_row(distance_threshold=0.5, is_percentile=False)
print(imputer.impute(row=0, column=0, data_array=data, mask_array=mask))
```

## Citing

If you use $N^2$, please cite [Chin et al. (2025)](https://arxiv.org/abs/2506.04166).
Machine-readable metadata is in [`CITATION.cff`](../CITATION.cff).
