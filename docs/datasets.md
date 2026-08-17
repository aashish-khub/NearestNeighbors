# Datasets ($N^2$-Bench)

$N^2$-Bench is a benchmark suite for matrix completion on realistic data. Rather than
synthetic low-rank matrices with uniform missingness, it collects four real-world tasks
whose missingness is structured, adaptive, or confounded, plus a configurable synthetic
generator for controlled experiments.

Every loader returns the same `(data, mask)` pair described in the
[Quickstart](quickstart.md#the-data-convention), so a method written against one dataset
runs on all of them.

## Common interface

```python
from nsquared import NNData, get_available_datasets

print(get_available_datasets())
# ['heartsteps', 'movielens', 'prompteval', 'prop99', 'synthetic_data']

NNData.help("heartsteps")                    # print this loader's parameters
NNData.get_data_params("heartsteps")         # the same, as a dict

loader = NNData.create("heartsteps", download=True, save_dir="./data")
data, mask = loader.process_data_scalar()            # (N, T),   (N, T)
data, mask = loader.process_data_distribution()      # (N, T, n), (N, T)
state = loader.get_full_state_as_dict(include_metadata=True)
```

**Arguments accepted by every loader**

| Argument | Default | Description |
| --- | --- | --- |
| `download` | `False` | Fetch the raw data on first use. Required the first time for the real datasets |
| `save_dir` | `"./"` | Where raw and processed data are cached |
| `agg` | `"mean"` | How multiple measurements collapse to a scalar. One of `mean`, `sum`, `median`, `std`, `variance` |
| `save_processed` | `False` | Persist the processed matrices to `save_dir` |

Downloads are cached, so repeated runs do not re-fetch.

Loaders differ in whether they blank out masked entries in the data matrix: `prompteval`
keeps the held-out truth in place, `synthetic_data` writes `np.nan`. Always drive
missingness from the mask — see
[User guide § The mask, and what it means](user_guide.md#the-mask-and-what-it-means).

---

## `heartsteps` — mobile health

HeartSteps V1, a micro-randomized trial of a mobile app that sends activity suggestions
([Klasnja et al., 2019](https://doi.org/10.1093/abm/kay067)). Rows are participants,
columns are decision points; an entry is the participant's step count in the hour after a
notification. Notifications were sent with probability 0.6 only at times the participant
was *available*, so missingness depends on latent daily routine — exactly the confounded
regime where classical methods struggle. The task is counterfactual: what would the step
count have been under the treatment that was not assigned?

Because raw step counts arrive at minute resolution, this dataset also supports the
distributional setting, with each entry holding the distribution of step counts within the
hour rather than just its mean.

| Parameter | Default | Description |
| --- | --- | --- |
| `freq` | `"5min"` | Resampling frequency for step count samples |
| `participants` | `37` | Number of participants |
| `max_study_day` | `52` | Maximum study day to include |
| `num_measurements` | `12` | Measurements per decision point (the `n` of the distributional setting) |

---

## `movielens` — recommender systems

The MovieLens ratings data ([Harper & Konstan, 2015](https://doi.org/10.1145/2827872)).
Rows are users, columns are movies, entries are ratings. Missingness is famously *not* at
random: users choose which films to watch and rate, so the observation pattern depends on
the ratings themselves.

This is the largest benchmark dataset and the experiments on it are memory-intensive; use
`sample_users` / `sample_movies` to work with a subsample.

| Parameter | Default | Description |
| --- | --- | --- |
| `sample_users` | `None` (all) | Number of users to subsample |
| `sample_movies` | `None` (all) | Number of movies to subsample |
| `seed` | `None` | Random seed for the subsample |

---

## `prop99` — causal panel data

California's Proposition 99 tobacco control program, the canonical synthetic control
dataset ([Abadie, Diamond & Hainmueller, 2010](https://doi.org/10.1198/jasa.2009.ap08746)).
Rows are US states, columns are years, entries are per-capita cigarette sales. The
counterfactual of interest is California's sales had the program not passed — a single
treated unit with a long pre-treatment period, which makes the missingness pattern block
structured rather than random.

| Parameter | Default | Description |
| --- | --- | --- |
| `start_year` | `1970` | First year included |
| `end_year` | `2019` | Last year included |
| `sample_states` | `None` (all) | Number of states to subsample |
| `seed` | `None` | Random seed for the subsample |

---

## `prompteval` — LLM evaluation

Multi-prompt LLM evaluation data ([Polo et al., 2024](https://arxiv.org/abs/2405.17202)).
Rows are models, columns are prompt templates (or template × example pairs), entries are
correctness scores. Evaluating every model on every prompt is expensive, so in practice
only a subset is run — matrix completion estimates the rest, and the distributional
setting captures the full score distribution across examples rather than only its mean.

| Parameter | Default | Description |
| --- | --- | --- |
| `tasks` | `None` (all) | Restrict to a list of tasks |
| `models` | `None` (all) | Restrict to a list of models |
| `propensity` | `None` | Proportion of entries to keep observed |
| `seed` | `None` | Random seed |

---

## `synthetic_data` — controlled simulation

A configurable generator for controlled experiments: latent factors of chosen
dimensionality, combined additively or multiplicatively, with Gaussian noise at a chosen
level. Useful for studying how methods scale with matrix size or degrade with noise. No
download is required.

> **Note:** only `mode="mcar"` is implemented. Passing `mode="mnar"` raises
> `NotImplementedError`.

| Parameter | Default | Description |
| --- | --- | --- |
| `num_rows` | `100` | $N$ |
| `num_cols` | `100` | $T$ |
| `seed` | `None` | Random seed |
| `stddev_noise` | `1` | Noise standard deviation |
| `snr` | `None` | Signal-to-noise ratio; overrides `stddev_noise` when set |
| `mode` | `"mcar"` | Missingness mechanism. Only `"mcar"` is implemented |
| `miss_prob` | `0.5` | Missingness probability (MCAR) |
| `mnar_deter` | `True` | Deterministic observation probabilities under MNAR (unused until MNAR lands) |
| `latent_factor_combination_model` | `"multiplicative"` | How row and column factors combine |
| `latent_factor_dimensionality` | `4` | Latent dimension $d$ |
| `rho` | `0.5` | Smoothness parameter of the nonlinear additive model |
| `simulated_data_nonlin_transform` | `""` | Optional nonlinear transform of the signal |

```python
from nsquared import NNData

loader = NNData.create(
    "synthetic_data", num_rows=200, num_cols=200, seed=0, snr=2.0, miss_prob=0.4,
)
data, mask = loader.process_data_scalar()
```

The generative model behind this loader lives in `nsquared.simulations` and can be
driven directly if you want the factors, the noiseless signal, and the mask separately
rather than a ready-made `(data, mask)` pair — see
[API reference § Simulations](api_reference.md#simulations).

---

## Pre-computed benchmark matrices

If you want to evaluate a method that does not fit the $N^2$ interface, the exact masked
matrices, unmasked ground truth, and masking matrices used in the paper are hosted
separately at [`calebchin/nsquared_bench_data`](https://github.com/calebchin/nsquared_bench_data).

## Adding a dataset

See [User guide § Add a benchmark dataset](user_guide.md#add-a-benchmark-dataset) and the
template in [`bench/README.md`](../bench/README.md#adding-new-datasets).
