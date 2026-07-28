# API reference

Everything listed here is importable directly from the top-level package:

```python
from nsquared import NearestNeighborImputer, row_row, Scalar, LeaveBlockOutValidation
```

Full parameter documentation lives in the docstrings; `help(nsquared.row_row)` in a REPL
gives the same content as this page in more detail.

---

## Core abstractions

### `NearestNeighborImputer(estimation_method, data_type, distance_threshold=None)`

The composite object users interact with. Pairs an estimator with an entry geometry.

| Parameter | Type | Description |
| --- | --- | --- |
| `estimation_method` | `EstimationMethod` | Which neighbors to use and how to weight them |
| `data_type` | `DataType` | The distance and average on entries |
| `distance_threshold` | `float`, `(float, float)`, or `None` | Threshold(s). If `None`, must be set by a `FitMethod` before imputing |

**Methods**

- `impute(row, column, data_array, mask_array, **kwargs) -> NDArray` — impute one entry.
  Raises `ValueError` if `distance_threshold` is `None`.
- `impute_all(data_array, mask_array) -> NDArray` — impute every entry of the matrix.
  Entries with no neighbor inside the threshold come back as `nan`.

### `DataType` (abstract)

Implement to support a new kind of entry. See
[User guide § Add a data type](user_guide.md#add-a-data-type).

- `distance(obj1, obj2) -> float`
- `average(object_list) -> Any`

### `EstimationMethod` (abstract)

Implement to add a new nearest neighbor variant.

- `impute(row, column, data_array, mask_array, distance_threshold, data_type, allow_self_neighbor=False, **kwargs) -> NDArray`
- `impute_all(data_array, mask_array, distance_threshold, data_type) -> NDArray` — has a
  default implementation that loops over `impute`; override it if your method can
  vectorise.

### `FitMethod` (abstract)

- `fit(data_array, mask_array, imputer, ret_trials=False)` — selects and sets
  `imputer.distance_threshold`.

---

## Data types

### `Scalar()`

Real-valued entries. `distance` is the squared difference; `average` is the arithmetic
mean (`nan`-aware).

### `DistributionKernelMMD(kernel, tuning_parameter=0.5, d=1)`

Distributional entries under maximum mean discrepancy geometry. Entries are `(n, d)`
sample arrays (1-D arrays are treated as `d = 1`).

| Parameter | Type | Description |
| --- | --- | --- |
| `kernel` | `str` | One of `"linear"`, `"square"`, `"exponential"` |
| `tuning_parameter` | `float` | Inverse bandwidth for the exponential kernel |
| `d` | `int` | Dimension of each sample |

`distance` is the unbiased U-statistic estimate of the squared MMD. `average` returns the
pooled samples of the selected neighbors — the empirical kernel barycenter — so the
returned array is longer than a single entry.

### `DistributionWassersteinSamples(num_samples)`

Distributional entries under 2-Wasserstein geometry, where every entry holds the same
number of samples. `distance` compares sorted samples; `average` is the quantile-averaged
barycenter and returns `num_samples` values.

### `DistributionWassersteinQuantile()`

Same geometry, but entries are represented by their empirical quantile functions rather
than raw samples. Use this when entries have differing sample counts.

---

## Estimation methods

All take `is_percentile: bool = True` unless noted, which interprets the distance
threshold as a quantile of observed distances rather than a raw distance.

| Class | Thresholds | Works with |
| --- | --- | --- |
| `RowRowEstimator(is_percentile=True)` | one | any `DataType` |
| `ColColEstimator(is_percentile=True)` | one | any `DataType` |
| `TSEstimator(is_percentile=True)` | `(row, col)` | any `DataType` |
| `DREstimator(is_percentile=True)` | `(row, col)` | `Scalar` only |
| `AWNNEstimator(delta=1, noise_variance=None, convergence_threshold=1e-4, max_iterations=10)` | none | any `DataType` |
| `AutoEstimator(is_percentile=True)` | `(row, col)` + `alpha` | `Scalar` only |

`DREstimator` and `AutoEstimator` subtract entries, which is undefined for probability
distributions, hence the `Scalar` restriction.

> `nsquared.nadaraya_watson.NadarayaWatsonEstimator` is present in the source tree but is
> not part of the public API: it does not yet implement the abstract
> `_calculate_distances` hook and cannot be instantiated.

---

## Constructors

Convenience functions that build the common scalar imputers.

| Function | Returns |
| --- | --- |
| `row_row(distance_threshold=None, is_percentile=True)` | `RowRowEstimator` + `Scalar` |
| `col_col(distance_threshold=None, is_percentile=True)` | `ColColEstimator` + `Scalar` |
| `ts_nn(distance_threshold_row=None, distance_threshold_col=None, is_percentile=True)` | `TSEstimator` + `Scalar` |
| `dr_nn(distance_threshold_row=None, distance_threshold_col=None, is_percentile=True)` | `DREstimator` + `Scalar` |
| `aw_nn(distance_threshold=0, delta=1, convergence_threshold=1e-4, max_iterations=10, noise_variance=None)` | `AWNNEstimator` + `Scalar` |

For a two-threshold constructor, passing only one of the two thresholds leaves the imputer
untuned (`distance_threshold` stays `None`), so a `FitMethod` must be run before imputing.

---

## Cross-validation

Every fit method holds out `block` — a list of `(row, column)` pairs that are **observed**
— imputes them, and minimises mean imputation error with `hyperopt`'s TPE search over
`n_trials` evaluations.

### `LeaveBlockOutValidation(block, distance_threshold_range, n_trials, data_type, allow_self_neighbor=False, rng=None)`

For single-threshold estimators. `fit(...) -> float`.

### `DualThresholdLeaveBlockOutValidation(block, distance_threshold_range_row, distance_threshold_range_col, n_trials, data_type, allow_self_neighbor=False)`

Abstract base for two-threshold estimators. Use one of:

- **`TSLeaveBlockOutValidation`** — for `ts_nn`. `fit(...) -> (row, col)`.
- **`DRLeaveBlockOutValidation`** — for `dr_nn`. `fit(...) -> (row, col)`.

### `AutoDRTSLeaveBlockOutValidation(block, distance_threshold_range_row, distance_threshold_range_col, alpha_range, n_trials, data_type, allow_self_neighbor=False)`

For `AutoEstimator`; additionally searches the mixing weight `alpha`.

### `evaluate_imputation(data_array, mask_array, imputer, test_cells, data_type, allow_self_neighbor=False, **kwargs) -> float`

The objective all the fit methods minimise: mean imputation error over `test_cells` under
`data_type.distance`. Useful directly if you want to score an imputer on a held-out set.

**Common arguments**

| Argument | Description |
| --- | --- |
| `block` | List of `(row, col)` tuples, drawn from positions where `mask == 1` |
| `*_range` | `(lower, upper)` bounds for the search |
| `n_trials` | Number of hyperopt evaluations |
| `allow_self_neighbor` | Whether an entry may serve as its own neighbor. Keep `False` for honest validation |
| `rng` | `numpy.random.Generator` for reproducible search (`LeaveBlockOutValidation` only) |

Pass `ret_trials=True` to `fit` to additionally receive the `hyperopt` `Trials` object.

---

## Datasets

### `NNData`

Factory for benchmark loaders.

- `NNData.create(dataset_name, download=False, save_dir="./", **kwargs) -> NNDataLoader`
- `NNData.get_data_params(dataset_name) -> dict` — the loader's dataset-specific
  parameters, as `{name: (type, default, description)}`.
- `NNData.help(dataset_name="")` — print available datasets, or one loader's parameters.

### `get_available_datasets() -> list[str]`

Names of all registered loaders. Imports every dataset subpackage so the list is complete;
loaders whose optional dependencies are missing are skipped.

### `register_dataset(name, params={})`

Class decorator that registers an `NNDataLoader` subclass under `name`.

### `NNDataLoader` (abstract)

Base class for loaders. Subclasses implement:

- `download_data()`
- `process_data_scalar(agg="mean") -> (data, mask)`
- `process_data_distribution(data_type=None) -> (data, mask)`
- `get_full_state_as_dict(include_metadata=False) -> dict`

See [Datasets](datasets.md) for the shipped loaders.

---

## Baselines

Not exported from the top level; import from `baselines`.

- `baselines.usvt` — universal singular value thresholding.
- `baselines.softimpute` — SoftImpute via `fancyimpute`.

## Utilities

- `nsquared.utils.experiments` — argument parsing shared by the benchmark scripts,
  including the `--estimation_method` alias registry.
- `nsquared.utils.plotting_utils` — per-method plot styling used by the `plot_*.py`
  scripts in [`examples/`](../examples/).
- `nsquared.utils.kernels`, `nsquared.utils.helper_fns` — kernel definitions and shared
  numerical helpers.
- `nsquared.simulations.mcar`, `nsquared.simulations.mnar` — missingness generators used
  by the synthetic data loader.
