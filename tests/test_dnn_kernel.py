"""Tests for the kernel MMD distributional data type.

``DistributionKernelMMD`` measures the distance between two entries with an
unbiased U-statistic estimate of the squared maximum mean discrepancy, and
averages them into an empirical kernel barycenter (the pooled samples of the
selected neighbors).
"""

import numpy as np
import pytest

from nsquared.data_types import DistributionKernelMMD
from nsquared.estimation_methods import RowRowEstimator, ColColEstimator, TSEstimator
from nsquared.nnimputer import EstimationMethod, NearestNeighborImputer

ROWS = 8
COLS = 8
SAMPLES = 30
KERNELS = ["linear", "square", "exponential"]


def test_unsupported_kernel_raises() -> None:
    """Only the documented kernels are accepted."""
    with pytest.raises(ValueError):
        DistributionKernelMMD(kernel="not-a-kernel")


@pytest.mark.parametrize("kernel", KERNELS)
def test_distance_is_nonnegative(kernel: str) -> None:
    """The squared MMD estimate is clipped at 0, so it is never negative."""
    data_type = DistributionKernelMMD(kernel=kernel)
    rng = np.random.default_rng(0)
    dist1 = rng.normal(size=SAMPLES)
    dist2 = rng.normal(loc=2.0, size=SAMPLES)

    assert data_type.distance(dist1, dist2) >= 0


@pytest.mark.parametrize("kernel", KERNELS)
def test_distance_is_symmetric(kernel: str) -> None:
    """The MMD does not depend on argument order."""
    data_type = DistributionKernelMMD(kernel=kernel)
    rng = np.random.default_rng(1)
    dist1 = rng.normal(size=SAMPLES)
    dist2 = rng.normal(loc=2.0, size=SAMPLES)

    assert data_type.distance(dist1, dist2) == pytest.approx(
        data_type.distance(dist2, dist1)
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_distance_is_invariant_to_sample_order(kernel: str) -> None:
    """Entries are unordered samples, so permuting them changes nothing."""
    data_type = DistributionKernelMMD(kernel=kernel)
    rng = np.random.default_rng(2)
    dist1 = rng.normal(size=SAMPLES)
    dist2 = rng.normal(loc=2.0, size=SAMPLES)

    assert data_type.distance(dist1, dist2) == pytest.approx(
        data_type.distance(rng.permutation(dist1), dist2)
    )


def test_distance_separates_distributions() -> None:
    """Samples from the same law are closer than samples from different laws."""
    data_type = DistributionKernelMMD(kernel="exponential", tuning_parameter=0.5)
    rng = np.random.default_rng(3)

    # Many samples so the U-statistic is a reliable estimate.
    same_law = data_type.distance(rng.normal(size=500), rng.normal(size=500))
    different_law = data_type.distance(
        rng.normal(size=500), rng.normal(loc=5.0, size=500)
    )

    assert different_law > same_law


def test_distance_of_identical_samples_is_small() -> None:
    """The unbiased estimate for two copies of one sample is near zero."""
    data_type = DistributionKernelMMD(kernel="exponential", tuning_parameter=0.5)
    dist = np.random.default_rng(4).normal(size=200)

    assert data_type.distance(dist, dist) == pytest.approx(0.0, abs=1e-8)


def test_average_pools_neighbor_samples() -> None:
    """The kernel barycenter is the concatenation of the neighbors' samples."""
    data_type = DistributionKernelMMD(kernel="linear")
    dists = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    np.testing.assert_allclose(data_type.average(dists), [1, 2, 3, 4, 5, 6])


def test_average_of_all_nan_entries_is_nan() -> None:
    """With no usable neighbors the average is a nan entry of the right shape."""
    data_type = DistributionKernelMMD(kernel="linear", d=1)
    dists = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    average = data_type.average(dists)
    assert average.shape == (1, 1)
    assert np.all(np.isnan(average))


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
    """A matrix whose entries all hold one distribution imputes to that distribution."""
    data_type = DistributionKernelMMD(kernel="exponential", tuning_parameter=0.5)
    imputer = NearestNeighborImputer(estimator, data_type, threshold)

    dist = np.random.default_rng(5).normal(size=SAMPLES)
    data = np.tile(dist, (ROWS, COLS, 1))
    mask = np.ones((ROWS, COLS), dtype=int)

    imputed = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)

    # The barycenter pools identical neighbors, so every pooled sample is one of
    # the original samples and the pooled mean matches the original mean.
    assert np.all(np.isin(imputed, dist))
    assert float(np.mean(imputed)) == pytest.approx(float(np.mean(dist)))


def test_imputation_recovers_entry_mean() -> None:
    """On a two-group signal, the imputed distribution's mean tracks the truth."""
    rng = np.random.default_rng(6)
    n_rows, n_cols = 25, 25

    row_effect = np.where(np.arange(n_rows) < n_rows // 2, -2.0, 2.0)
    means = row_effect[:, None] * np.ones((1, n_cols))
    data = rng.normal(loc=means[:, :, None], scale=0.1, size=(n_rows, n_cols, SAMPLES))
    mask = np.ones((n_rows, n_cols), dtype=int)
    mask[0, 0] = 0

    # A percentile threshold of 0.25 keeps well inside the same-group rows.
    imputer = NearestNeighborImputer(
        RowRowEstimator(is_percentile=True),
        DistributionKernelMMD(kernel="exponential", tuning_parameter=0.5),
        distance_threshold=0.25,
    )
    imputed = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)

    assert float(np.mean(imputed)) == pytest.approx(means[0, 0], abs=0.2)


if __name__ == "__main__":
    pytest.main()
