# Quickstart

Every example on this page is self-contained and runs in a few seconds on a laptop.

## The data convention

$N^2$ works with two arrays that always come as a pair:

- **`data_array`** — the (partially observed) matrix. For scalar problems this has shape
  `(N, T)`. For distributional problems it has shape `(N, T, n)`, where each entry holds
  `n` samples from that entry's distribution.
- **`mask_array`** — an integer `(N, T)` array where `1` means "entry observed" and `0`
  means "entry missing". Values of `data_array` at masked-out positions are ignored.

Every benchmark loader returns exactly this pair.

> **The mask is authoritative, not `np.isnan(data)`.** Some loaders deliberately keep the
> true values at masked-out positions so they can be scored against. See
> [User guide § The mask, and what it means](user_guide.md#the-mask-and-what-it-means).

## Scalar matrix completion

```python
import numpy as np
from nsquared import NNData, row_row, Scalar, LeaveBlockOutValidation

# 1. Load a partially observed matrix.
loader = NNData.create("synthetic_data", num_rows=50, num_cols=50, seed=0, miss_prob=0.3)
data, mask = loader.process_data_scalar()

# 2. Pick an estimator.
imputer = row_row()

# 3. Tune the distance threshold on a held-out block of *observed* entries.
rng = np.random.default_rng(0)
observed = np.argwhere(mask == 1)
block = [tuple(map(int, observed[i])) for i in rng.choice(len(observed), 30, replace=False)]

cv = LeaveBlockOutValidation(
    block=block,
    distance_threshold_range=(0, 1),
    n_trials=20,
    data_type=Scalar(),
)
best_threshold = cv.fit(data, mask, imputer)

# 4. Impute a single entry.
value = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
```

`cv.fit` sets `imputer.distance_threshold` in place and also returns it. Calling `impute`
before a threshold is set raises `ValueError`.

### Skipping cross-validation

If you already know the threshold, pass it to the constructor:

```python
imputer = row_row(distance_threshold=0.5, is_percentile=False)
value = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
```

`is_percentile=True` (the default) interprets the threshold as a *quantile* of the
observed pairwise distances, which makes the same number transferable across datasets on
different scales. `is_percentile=False` interprets it as a raw distance.

### Imputing the whole matrix

```python
imputed = imputer.impute_all(data, mask)   # shape (N, T)
```

Entries with no neighbor inside the threshold come back as `nan`. If you get more `nan`s
than you expect, raise the threshold or use a percentile threshold.

## Switching methods

This is the point of the package — the estimator is the only thing that changes.

```python
from nsquared import row_row, col_col, ts_nn, dr_nn, aw_nn

imputer = row_row()                                              # row-wise NN
imputer = col_col()                                              # column-wise NN
imputer = ts_nn(distance_threshold_row=0.5, distance_threshold_col=0.5)   # two-sided NN
imputer = dr_nn(distance_threshold_row=0.5, distance_threshold_col=0.5)   # doubly robust NN
imputer = aw_nn(noise_variance=1.0)                              # adaptively-weighted NN
```

Two-sided and doubly robust NN take *two* thresholds, so they need the dual-threshold
cross-validators:

```python
from nsquared import ts_nn, TSLeaveBlockOutValidation, DRLeaveBlockOutValidation, Scalar

imputer = ts_nn()
cv = TSLeaveBlockOutValidation(
    block=block,
    distance_threshold_range_row=(0, 1),
    distance_threshold_range_col=(0, 1),
    n_trials=20,
    data_type=Scalar(),
)
row_threshold, col_threshold = cv.fit(data, mask, imputer)
```

Auto NN, which balances a doubly robust and a two-sided estimate, additionally tunes a
mixing weight:

```python
from nsquared import (
    NearestNeighborImputer, AutoEstimator, AutoDRTSLeaveBlockOutValidation, Scalar,
)

imputer = NearestNeighborImputer(AutoEstimator(), Scalar())
cv = AutoDRTSLeaveBlockOutValidation(
    block=block,
    distance_threshold_range_row=(0, 1),
    distance_threshold_range_col=(0, 1),
    alpha_range=(0, 1),
    n_trials=20,
    data_type=Scalar(),
)
cv.fit(data, mask, imputer)
```

## Distributional matrix completion

In the distributional setting each entry is an empirical distribution — for example, the
60 one-minute step counts inside an hour, rather than their mean. $N^2$ imputes the whole
distribution.

The only change is the `DataType`: the estimators are identical.

```python
import numpy as np
from nsquared import (
    NearestNeighborImputer, RowRowEstimator,
    DistributionWassersteinSamples, DistributionKernelMMD,
)

# data has shape (N, T, n): n samples per entry.
rng = np.random.default_rng(0)
N, T, n = 20, 20, 50
means = rng.normal(size=(N, 1)) @ rng.normal(size=(T, 1)).T
data = rng.normal(loc=means[:, :, None], scale=1.0, size=(N, T, n))
mask = (rng.random((N, T)) < 0.8).astype(int)

# Wasserstein-2 geometry: neighbors matched by W2 distance, averaged by W2 barycenter.
imputer = NearestNeighborImputer(
    RowRowEstimator(),
    DistributionWassersteinSamples(num_samples=n),
    distance_threshold=0.9,
)
distribution = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
print(distribution.shape)     # (50,) — an imputed empirical distribution

# Kernel MMD geometry instead: same estimator, different DataType.
imputer = NearestNeighborImputer(
    RowRowEstimator(),
    DistributionKernelMMD(kernel="exponential", tuning_parameter=0.5),
    distance_threshold=0.9,
)
distribution = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
```

The two geometries return different representations of the imputed distribution:
`DistributionWassersteinSamples` returns the quantile-averaged barycenter as `n` sorted
samples, while `DistributionKernelMMD` returns the pooled samples of the selected
neighbors, which is the empirical kernel barycenter. Both can be summarised with
`np.mean(...)` to recover a scalar estimate, which is how the two settings are compared in
the benchmark.

To compare against a scalar method, take the mean of the imputed distribution:

```python
scalar_estimate = float(np.mean(distribution))
```

Any estimator works with any data type, with one exception: `DREstimator` (doubly robust)
needs a well-defined subtraction on entries and therefore only supports `Scalar`.

## Using a real benchmark dataset

```python
from nsquared import NNData, get_available_datasets

print(get_available_datasets())
# ['heartsteps', 'movielens', 'prompteval', 'prop99', 'synthetic_data']

NNData.help("heartsteps")        # prints the loader's parameters

loader = NNData.create("heartsteps", download=True, save_dir="./data")
data, mask = loader.process_data_scalar()               # scalar setting
data, mask = loader.process_data_distribution()         # distributional setting
```

`download=True` fetches the raw data on first use and caches it under `save_dir`. See
[Datasets](datasets.md) for what each matrix contains and which options each loader takes,
and [`examples/`](../examples/) for complete evaluation scripts per dataset.

## Where next

- [User guide](user_guide.md) — how the pieces fit together, and how to add your own
  estimator or data type.
- [API reference](api_reference.md) — every public class and function.
- [`bench/README.md`](../bench/README.md) — reproducing the paper's experiments.
