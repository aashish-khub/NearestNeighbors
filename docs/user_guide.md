# User guide

This page explains how $N^2$ is put together and how to extend it. If you just want to run
something, start with the [Quickstart](quickstart.md).

## The model

Write the data as an $N \times T$ matrix in which entry $(i, t)$ is observed only when
$A_{i,t} = 1$:

$$
Z_{i,t} = \begin{cases}
  X_1(i,t), \dots, X_n(i,t) \sim \mu_{i,t} & \text{if } A_{i,t} = 1, \\
  \text{unknown} & \text{if } A_{i,t} = 0.
\end{cases}
$$

- When $n = 1$ this is **scalar matrix completion**: the target is the mean
  $\theta_{i,t} = \int x \, d\mu_{i,t}(x)$.
- When $n \ge 2$ this is **distributional matrix completion**: the target is the whole
  distribution $\mu_{i,t}$.

$N^2$ treats these as the same problem with a different notion of "what lives in a cell".

## The mask, and what it means

Every loader and every estimator takes two arrays: a data matrix and a `mask_array`. Two
things about the mask are easy to get wrong.

### The mask encodes treatment, not merely "is this cell populated"

In the counterfactual-inference framing, `mask[i, t] == 1` means *entry `(i, t)` was
observed **under the treatment you are asking about***. It is an availability indicator
for one arm of the experiment, not a general "data exists here" flag.

HeartSteps makes this concrete. A notification either was or was not sent at decision
point `t` for participant `i`, and the loader derives the mask from that treatment
indicator (`send.sedentary`). The estimand is the counterfactual: what the step count
*would* have been under the treatment that was not assigned. So `mask[i, t] == 0` does not
mean "we know nothing about this participant at this time" — it means "this participant
was not treated here, so their outcome under treatment is the thing we are imputing".

The practical consequence: to study the other arm, you re-derive the mask, you do not
reuse the same one. And a mask built by asking "which cells are non-empty?" is generally
the *wrong* mask for a causal question, because it conflates two different arms.

### `mask == 0` does not imply `nan` in the data matrix

The mask is authoritative; the data matrix is not. Loaders differ deliberately:

- `synthetic_data` **does** write `np.nan` at masked-out positions, because the noiseless
  ground truth is returned separately via `get_full_state_as_dict()`.
- `prompteval` **deliberately leaves the true values in place** at `mask == 0`. Its loader
  contains the line `# data[mask == 0] = np.nan`, commented out on purpose, so that the
  held-out entries remain available for scoring.

So do not infer missingness with `np.isnan(data)`, and do not assume masked entries are
safe to read. Estimators consult `mask_array`, and reading `data_array` where
`mask_array == 0` may hand you the answer you are trying to predict — which silently
turns a benchmark into a leak.

```python
# Wrong: the data matrix may still hold the held-out truth.
missing = np.isnan(data)

# Right: the mask is the single source of truth.
missing = mask == 0
```

Conversely `nan` can appear where `mask == 1`, since an entry can be observed but
genuinely undefined, so `mask == 1` is not a promise that the value is finite either.

## Two modules, two abstractions

Every nearest neighbor variant in the literature can be written as a composition of two
operations:

**Distance** — how far apart are two rows (or two columns)? Row and column distances are
averages of an entry-wise distance $\hat\varphi$ over the positions where both entries were
observed:

$$
\rho^{\text{row}}_{i,j} = \frac{\sum_{s \neq t} A_{i,s} A_{j,s}\, \hat\varphi(Z_{i,s}, Z_{j,s})}{\sum_{s \neq t} A_{i,s} A_{j,s}}
$$

**Average** — given weights $w_{j,s}$ over the observed entries, what is the best single
estimate? This is a weighted Fréchet mean under the entry metric $\varphi$:

$$
\hat\theta = \arg\min_{x} \sum_{j,s} w_{j,s} A_{j,s}\, \varphi(x, Z_{j,s})
$$

$N^2$ maps these onto two abstract base classes, which vary *independently*:

| Abstraction | Answers | Where it lives |
| --- | --- | --- |
| `DataType` | What is $\varphi$? How do I average? | `nsquared/data_types.py` |
| `EstimationMethod` | Which neighbors, with what weights, composed how? | `nsquared/estimation_methods.py` |

This is the key design decision. A `DataType` knows the *geometry* of an entry and nothing
about neighbors; an `EstimationMethod` knows the *estimator* and nothing about what an
entry contains. That independence is what makes the two axes multiply: adding one new
`DataType` gives you every estimator for it, and adding one new estimator gives you
scalars and all distributional geometries at once.

### `DataType`

Two methods:

```python
class DataType(ABC):
    @abstractmethod
    def distance(self, obj1: Any, obj2: Any) -> float: ...

    @abstractmethod
    def average(self, object_list: npt.NDArray[Any]) -> Any: ...
```

Shipped implementations:

| Class | Entry | `distance` | `average` |
| --- | --- | --- | --- |
| `Scalar` | a float | squared difference | arithmetic mean |
| `DistributionKernelMMD` | `n` samples | U-statistic of squared MMD | empirical kernel barycenter |
| `DistributionWassersteinSamples` | `n` samples | squared $W_2$ via sorted samples | quantile-averaged barycenter |
| `DistributionWassersteinQuantile` | a quantile function | squared $W_2$ on quantiles | pointwise mean of quantile functions |

### `EstimationMethod`

```python
class EstimationMethod(ABC):
    @abstractmethod
    def impute(self, row, column, data_array, mask_array,
               distance_threshold, data_type, allow_self_neighbor=False, **kwargs): ...
```

Shipped implementations:

| Class | Idea | Thresholds |
| --- | --- | --- |
| `RowRowEstimator` | average over rows similar to row $i$ | one |
| `ColColEstimator` | average over columns similar to column $t$ | one |
| `TSEstimator` | two-sided: use both row and column neighbors | two |
| `DREstimator` | doubly robust: combine row and column estimates, subtracting the overlap | two |
| `AWNNEstimator` | adaptively-weighted: solve for continuous weights instead of a hard cutoff | none (uses `delta`, `noise_variance`) |
| `AutoEstimator` | convex combination of the doubly robust and two-sided estimates | two + mixing weight |
| `NadarayaWatsonEstimator` | kernel-smoothed: weight every row by a kernel of its distance instead of applying a cutoff | one (kernel bandwidth) |

`DREstimator` is the one estimator that is *not* geometry-agnostic: it subtracts entries,
which is undefined in the space of probability distributions, so it requires `Scalar`.
`NadarayaWatsonEstimator` is restricted the same way, since a kernel-weighted mean of the
target column is only defined for scalar entries.

### Composing them

`NearestNeighborImputer` is the composite of the two, an instance of the *Composite*
design pattern:

```python
from nsquared import NearestNeighborImputer, TSEstimator, Scalar

imputer = NearestNeighborImputer(
    estimation_method=TSEstimator(),
    data_type=Scalar(),
    distance_threshold=(0.5, 0.5),
)
imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
```

The `row_row`, `col_col`, `ts_nn`, `dr_nn`, and `aw_nn` functions are thin constructors
that build the common scalar combinations for you.

## Thresholds and tuning

Nearest neighbor estimators have one or two distance thresholds: an entry $j$ counts as a
neighbor of $i$ when $\rho^{\text{row}}_{i,j}$ falls below the threshold.

`is_percentile=True` (the default on most estimators) interprets the threshold as a
quantile of the observed distances rather than a raw distance. This makes a given value
comparable across datasets with different scales, and is usually what you want.

Tuning lives in a separate `FitMethod` hierarchy so that cross-validation strategies and
estimators can be mixed freely. Each holds out a block of *observed* entries, imputes
them, and minimises the imputation error with `hyperopt`'s TPE search:

| Class | For estimators with |
| --- | --- |
| `LeaveBlockOutValidation` | one threshold (`row_row`, `col_col`, Nadaraya–Watson) |
| `TSLeaveBlockOutValidation` | two thresholds (`ts_nn`) |
| `DRLeaveBlockOutValidation` | two thresholds (`dr_nn`) |
| `AutoDRTSLeaveBlockOutValidation` | two thresholds plus a mixing weight (`AutoEstimator`) |

All of them mutate `imputer.distance_threshold` in place and return the chosen value(s).
Pass `ret_trials=True` to also get back the `hyperopt` `Trials` object for inspection.

## Extending $N^2$

### Add a data type

Implement `distance` and `average` for your entry space. Everything else follows.

```python
import numpy as np
import numpy.typing as npt
from typing import Any
from nsquared import DataType


class L1Scalar(DataType):
    """Scalars under absolute rather than squared error."""

    def distance(self, obj1: float, obj2: float) -> float:
        return float(abs(obj1 - obj2))

    def average(self, object_list: npt.NDArray[Any]) -> Any:
        # The Fréchet mean under |.| is the median.
        return float(np.nanmedian(object_list))
```

That class immediately works with `RowRowEstimator`, `TSEstimator`, `AWNNEstimator`, and
the cross-validators — nothing else needs to change. The same route is how you would add
entries that are text embeddings or images: define a distance and a barycenter, and every
estimator comes along for free.

### Add an estimator

Subclass `EstimationMethod` in `src/nsquared/estimation_methods.py` and implement `impute`
(and `_calculate_distances`). Write it against the abstract `DataType` interface — call
`data_type.distance(...)` and `data_type.average(...)` rather than `np.mean` — so your
estimator works for distributions as well as scalars.

If your method needs a tuning strategy that does not exist yet, add a `FitMethod` in
`src/nsquared/fit_methods.py`.

To make the method usable from the benchmark scripts, register an alias for it in
`src/nsquared/utils/experiments.py` (the `--estimation_method` parser argument) and add
plot settings in `src/nsquared/utils/plotting_utils.py`.

### Add a benchmark dataset

Create `src/nsquared/datasets/<name>/loader.py` with a subclass of `NNDataLoader`
decorated with `@register_dataset("<name>", params)`, implementing `download_data`,
`process_data_scalar`, `process_data_distribution`, and `get_full_state_as_dict`. Add an
`__init__.py` that re-exports the class. The loader is then reachable through
`NNData.create("<name>")` and is discovered automatically by `get_available_datasets()`.

Full templates for all three extension points are in
[`bench/README.md`](../bench/README.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Baselines

For comparison, `nsquared.baselines` packages two classical non-NN matrix completion
methods:

- `nsquared.baselines.usvt` — universal singular value thresholding (Chatterjee, 2015).
- `nsquared.baselines.softimpute` — SoftImpute (Mazumder et al., 2010; Hastie et al.,
  2015).
- `nsquared.baselines.knn_impute` — k-nearest-neighbor imputation, matching
  `sklearn.impute.KNNImputer`. Note this is the *feature-matrix* form of k-NN imputation,
  matching on rows only; it is the thing the row-wise estimators in this package
  generalize, and is included so that generalization can be measured.
- `nsquared.baselines.knn_impute_columnwise` — the same over columns.

Both take a matrix with `np.nan` in the missing positions, so they drop into the same
evaluation loop, and both are implemented directly on NumPy — no optional dependency is
needed for either.

## Further reading

The methods implemented here are described in:

- Li, Shah, Song & Yu (2019), *Nearest neighbors for matrix estimation interpreted as
  blind regression for latent variable model*.
- Dwivedi et al. (2022), *Counterfactual inference for sequential experiments* and *Doubly
  robust nearest neighbors in factor models*.
- Sadhukhan, Paul & Dwivedi (2024, 2025), two-sided and adaptively-weighted NN.
- Choi et al. (2024) and Feitelberg et al. (2024), kernel and Wasserstein distributional NN.
- Chin et al. (2025), the $N^2$ package and benchmark.

Full citations are in the [README](../README.md#citation).
