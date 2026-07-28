"""Tests for the Wasserstein distributional data types.

Covers ``DistributionWassersteinSamples`` and ``DistributionWassersteinQuantile``
both as standalone geometries (distance / average) and composed with the
estimators into a ``NearestNeighborImputer``.
"""

import numpy as np
import pytest

from nsquared.data_types import (
    DistributionWassersteinSamples,
    DistributionWassersteinQuantile,
)
from nsquared.estimation_methods import RowRowEstimator, ColColEstimator, TSEstimator
from nsquared.nnimputer import EstimationMethod, NearestNeighborImputer

ROWS = 8
COLS = 8
SAMPLES = 20


def test_distance_of_identical_distributions_is_zero() -> None:
    """The distance between a distribution and itself is 0."""
    data_type = DistributionWassersteinSamples(num_samples=SAMPLES)
    rng = np.random.default_rng(0)
    dist = rng.normal(size=SAMPLES)

    assert data_type.distance(dist, dist) == pytest.approx(0.0)


def test_distance_is_symmetric() -> None:
    """The Wasserstein distance does not depend on argument order."""
    data_type = DistributionWassersteinSamples(num_samples=SAMPLES)
    rng = np.random.default_rng(1)
    dist1 = rng.normal(size=SAMPLES)
    dist2 = rng.normal(loc=3.0, size=SAMPLES)

    assert data_type.distance(dist1, dist2) == pytest.approx(
        data_type.distance(dist2, dist1)
    )


def test_distance_is_invariant_to_sample_order() -> None:
    """Entries are empirical distributions, so sample order carries no information."""
    data_type = DistributionWassersteinSamples(num_samples=SAMPLES)
    rng = np.random.default_rng(2)
    dist1 = rng.normal(size=SAMPLES)
    dist2 = rng.normal(size=SAMPLES)

    shuffled = rng.permutation(dist1)
    assert data_type.distance(dist1, dist2) == pytest.approx(
        data_type.distance(shuffled, dist2)
    )


def test_distance_grows_with_separation() -> None:
    """Shifting one distribution further away increases the distance."""
    data_type = DistributionWassersteinSamples(num_samples=SAMPLES)
    base = np.linspace(0, 1, SAMPLES)

    near = data_type.distance(base, base + 1.0)
    far = data_type.distance(base, base + 5.0)
    assert far > near


def test_mismatched_sample_counts_raise() -> None:
    """Comparing distributions with different sample counts is an error."""
    data_type = DistributionWassersteinSamples(num_samples=SAMPLES)

    with pytest.raises(ValueError):
        data_type.distance(np.zeros(SAMPLES), np.zeros(SAMPLES + 1))


def test_average_of_identical_distributions() -> None:
    """The barycenter of copies of one distribution is that distribution."""
    data_type = DistributionWassersteinSamples(num_samples=SAMPLES)
    dist = np.sort(np.random.default_rng(3).normal(size=SAMPLES))

    average = data_type.average(np.array([dist] * 5))
    np.testing.assert_allclose(average, dist)


def test_average_is_the_quantile_mean() -> None:
    """The W2 barycenter averages sorted samples pointwise."""
    data_type = DistributionWassersteinSamples(num_samples=3)
    dists = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])

    # Sorted: [1, 2, 3] and [4, 5, 6] -> pointwise mean [2.5, 3.5, 4.5].
    np.testing.assert_allclose(data_type.average(dists), [2.5, 3.5, 4.5])


@pytest.mark.parametrize(
    "estimator,threshold",
    [
        (RowRowEstimator(is_percentile=False), 1.0),
        (ColColEstimator(is_percentile=False), 1.0),
        (TSEstimator(is_percentile=False), (1.0, 1.0)),
    ],
)
def test_constant_matrix_imputation(
    estimator: EstimationMethod, threshold: float | tuple[float, float]
) -> None:
    """Every entry holding the same distribution imputes to that distribution."""
    data_type = DistributionWassersteinSamples(num_samples=SAMPLES)
    imputer = NearestNeighborImputer(estimator, data_type, threshold)

    dist = np.sort(np.random.default_rng(4).normal(size=SAMPLES))
    data = np.tile(dist, (ROWS, COLS, 1))
    mask = np.ones((ROWS, COLS), dtype=int)

    imputed = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
    np.testing.assert_allclose(imputed, dist)


def test_imputation_recovers_entry_mean() -> None:
    """On a two-group signal, the imputed distribution's mean tracks the truth."""
    rng = np.random.default_rng(5)
    n_rows, n_cols = 25, 25

    # Rows come in two groups, so each row has genuine neighbors.
    row_effect = np.where(np.arange(n_rows) < n_rows // 2, -2.0, 2.0)
    means = row_effect[:, None] * np.ones((1, n_cols))
    data = rng.normal(loc=means[:, :, None], scale=0.1, size=(n_rows, n_cols, SAMPLES))
    mask = np.ones((n_rows, n_cols), dtype=int)
    mask[0, 0] = 0

    # Within-group row distances are ~0.05 here and cross-group ones ~320, so a
    # raw threshold of 1.0 selects exactly the rows in the same group.
    imputer = NearestNeighborImputer(
        RowRowEstimator(is_percentile=False),
        DistributionWassersteinSamples(num_samples=SAMPLES),
        distance_threshold=1.0,
    )
    imputed = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)

    assert float(np.mean(imputed)) == pytest.approx(means[0, 0], abs=0.2)


def test_quantile_data_type_distance_and_average() -> None:
    """The quantile-function geometry gives a symmetric distance."""
    data_type = DistributionWassersteinQuantile()
    rng = np.random.default_rng(6)

    # This data type consumes quantile *functions*, not raw samples.
    quantile1 = data_type.empirical_quantile_function(np.sort(rng.normal(size=SAMPLES)))
    quantile2 = data_type.empirical_quantile_function(
        np.sort(rng.normal(loc=1.0, size=SAMPLES))
    )

    assert data_type.distance(quantile1, quantile1) == pytest.approx(0.0)
    assert data_type.distance(quantile1, quantile2) == pytest.approx(
        data_type.distance(quantile2, quantile1)
    )
    assert data_type.distance(quantile1, quantile2) > 0

    average = data_type.average(np.array([quantile1, quantile1], dtype=object))
    grid = np.linspace(0, 1, 50)
    np.testing.assert_allclose(average(grid), quantile1(grid))


def test_quantile_average_of_shifted_distributions() -> None:
    """Averaging quantile functions of shifted copies shifts the barycenter."""
    data_type = DistributionWassersteinQuantile()
    samples = np.sort(np.random.default_rng(7).normal(size=SAMPLES))

    quantile = data_type.empirical_quantile_function(samples)
    shifted = data_type.empirical_quantile_function(samples + 4.0)

    average = data_type.average(np.array([quantile, shifted], dtype=object))
    grid = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(average(grid), quantile(grid) + 2.0)


if __name__ == "__main__":
    pytest.main()
