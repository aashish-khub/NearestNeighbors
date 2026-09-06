"""Tests for the Nadaraya-Watson estimator."""

import numpy as np
import pytest

from nsquared.data_types import Scalar
from nsquared.estimation_methods import RowRowEstimator
from nsquared.nadaraya_watson import NadarayaWatsonEstimator
from nsquared.nnimputer import NearestNeighborImputer

# Rows 0 and 1 agree on every non-target column, so they sit at distance 0 from
# each other; rows 2 and 3 sit at distance 1 and 4 from row 0 respectively.
DATA = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
        [2.0, 2.0, 3.0],
    ]
)
MASK = np.ones((4, 3), dtype=bool)
# Distances from row 0 to each row, averaged over the non-target columns 0 and 1.
EXPECTED_ROW_DISTS = np.array([0.0, 0.0, 1.0, 4.0])
TARGET_COLUMN = 2


def test_estimator_is_concrete() -> None:
    """The estimator implements every abstract hook and can be instantiated.

    Regression test: NadarayaWatsonEstimator previously omitted
    EstimationMethod._calculate_distances, so construction raised TypeError.
    """
    estimator = NadarayaWatsonEstimator()
    assert str(estimator) == "NadarayaWatsonEstimator(kernel=gaussian)"


@pytest.mark.parametrize("kernel", NadarayaWatsonEstimator.valid_kernels)
def test_every_valid_kernel_constructs(kernel: str) -> None:
    """Each advertised kernel can be selected."""
    estimator = NadarayaWatsonEstimator(kernel=kernel)
    assert estimator.kernel == kernel


def test_invalid_kernel_raises_value_error() -> None:
    """An unknown kernel is rejected with ValueError.

    Regression test: the validation used to read self.kernel before it was
    assigned, which raised AttributeError instead.
    """
    with pytest.raises(ValueError, match="is not a valid kernel"):
        NadarayaWatsonEstimator(kernel="not_a_kernel")


def test_distances_match_row_row_estimator() -> None:
    """Distances are the row-row distances, normalized once over the overlap.

    Regression test: the normalization used to sit inside the per-column loop,
    so the running total was divided by the overlap count once per column.
    """
    nw = NadarayaWatsonEstimator(is_percentile=False)
    row_row = RowRowEstimator(is_percentile=False)

    nw._calculate_distances(0, TARGET_COLUMN, DATA, MASK, Scalar())
    row_row._calculate_distances(0, TARGET_COLUMN, DATA, MASK, Scalar())

    np.testing.assert_allclose(nw.row_distances[0], row_row.row_distances[0])

    # Excluding the target column, these are the mean squared distances to row 0.
    all_dists = np.copy(nw.row_distances[0])
    all_dists[:, TARGET_COLUMN] = np.nan
    np.testing.assert_allclose(np.nanmean(all_dists, axis=1), EXPECTED_ROW_DISTS)


def test_distances_are_cached() -> None:
    """Repeated calls for the same row reuse the cached distance matrix."""
    estimator = NadarayaWatsonEstimator(is_percentile=False)
    estimator._calculate_distances(0, TARGET_COLUMN, DATA, MASK, Scalar())
    cached = estimator.row_distances[0]
    estimator._calculate_distances(0, TARGET_COLUMN, DATA, MASK, Scalar())
    assert estimator.row_distances[0] is cached


def test_gaussian_matches_hand_computed_weights() -> None:
    """The estimate is the kernel-weighted mean of the observed target column."""
    eta = 1.0
    estimator = NadarayaWatsonEstimator(kernel="gaussian", is_percentile=False)

    estimated_value = estimator.impute(
        row=0,
        column=TARGET_COLUMN,
        data_array=DATA,
        mask_array=MASK,
        distance_threshold=eta,
        data_type=Scalar(),
    )

    # Gaussian kernel, written out from its definition rather than reusing the
    # implementation under test.
    weights = np.exp(-0.5 * EXPECTED_ROW_DISTS / eta**2)
    expected = weights @ DATA[:, TARGET_COLUMN] / weights.sum()
    assert np.isclose(estimated_value, expected)


def test_unobserved_target_entries_are_excluded() -> None:
    """Rows whose target entry is missing contribute neither weight nor value.

    Regression test: the estimate used to be taken over the raw target column,
    so whatever sat behind the mask was averaged in.
    """
    eta = 1.0
    data = np.copy(DATA)
    mask = np.copy(MASK)
    # Row 3's target entry is unobserved, and the array holds garbage there.
    data[3, TARGET_COLUMN] = 1e6
    mask[3, TARGET_COLUMN] = False

    estimator = NadarayaWatsonEstimator(kernel="gaussian", is_percentile=False)
    estimated_value = estimator.impute(
        row=0,
        column=TARGET_COLUMN,
        data_array=data,
        mask_array=mask,
        distance_threshold=eta,
        data_type=Scalar(),
    )

    weights = np.exp(-0.5 * EXPECTED_ROW_DISTS[:3] / eta**2)
    expected = weights @ DATA[:3, TARGET_COLUMN] / weights.sum()
    assert np.isclose(estimated_value, expected)
    # The masked value is six orders of magnitude away; nothing near it survives.
    assert estimated_value < 10


def test_box_kernel_agrees_with_row_row_estimator() -> None:
    """A box kernel reduces the NW estimate to the vanilla row-row estimate.

    Both average the observed target entries of the rows within the bandwidth,
    so on the same input they must agree exactly.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(size=(12, 8))
    mask = rng.random((12, 8)) < 0.8
    eta = 3.0

    nw = NadarayaWatsonEstimator(kernel="box", is_percentile=False)
    row_row = RowRowEstimator(is_percentile=False)

    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            nw_value = nw.impute(
                row, column, data, mask, eta, Scalar(), allow_self_neighbor=False
            )
            row_row_value = row_row.impute(
                row, column, data, mask, eta, Scalar(), allow_self_neighbor=False
            )
            if np.isnan(row_row_value):
                assert np.isnan(nw_value), f"({row}, {column})"
            else:
                assert np.isclose(nw_value, row_row_value), f"({row}, {column})"


def test_constant_matrix_imputes_the_constant() -> None:
    """Sanity check: a constant matrix imputes to that constant."""
    data = np.full((6, 5), 0.5)
    mask = np.ones((6, 5), dtype=bool)

    for kernel in NadarayaWatsonEstimator.valid_kernels:
        estimator = NadarayaWatsonEstimator(kernel=kernel, is_percentile=False)
        for r, c in np.ndindex(data.shape):
            estimated_value = estimator.impute(
                row=r,
                column=c,
                data_array=data,
                mask_array=mask,
                distance_threshold=1.0,
                data_type=Scalar(),
            )
            assert np.isclose(estimated_value, 0.5), kernel


def test_percentile_threshold_uses_quantile_bandwidth() -> None:
    """With is_percentile, the threshold selects a quantile of the distances."""
    quantile = 0.75
    estimator = NadarayaWatsonEstimator(kernel="gaussian", is_percentile=True)

    estimated_value = estimator.impute(
        row=0,
        column=TARGET_COLUMN,
        data_array=DATA,
        mask_array=MASK,
        distance_threshold=quantile,
        data_type=Scalar(),
    )

    eta = np.quantile(EXPECTED_ROW_DISTS, quantile)
    weights = np.exp(-0.5 * EXPECTED_ROW_DISTS / eta**2)
    expected = weights @ DATA[:, TARGET_COLUMN] / weights.sum()
    assert np.isclose(estimated_value, expected)


def test_zero_bandwidth_falls_back_to_exact_matches() -> None:
    """A zero bandwidth is the limit where only distance-zero rows contribute."""
    estimator = NadarayaWatsonEstimator(kernel="gaussian", is_percentile=False)
    estimated_value = estimator.impute(
        row=0,
        column=TARGET_COLUMN,
        data_array=DATA,
        mask_array=MASK,
        distance_threshold=0.0,
        data_type=Scalar(),
    )
    # Rows 0 and 1 are the distance-zero rows; their target entries are 0 and 1.
    assert np.isclose(estimated_value, 0.5)


def test_singular_box_returns_exact_match_value() -> None:
    """The singular kernel resolves its singularity to the exact-match outcome."""
    estimator = NadarayaWatsonEstimator(kernel="singular_box", is_percentile=False)
    estimated_value = estimator.impute(
        row=0,
        column=TARGET_COLUMN,
        data_array=DATA,
        mask_array=MASK,
        distance_threshold=1.0,
        data_type=Scalar(),
    )
    assert np.isclose(estimated_value, 0.5)


def test_returns_nan_when_target_column_is_unobserved() -> None:
    """With no observed target entry there is nothing to average."""
    mask = np.copy(MASK)
    mask[:, TARGET_COLUMN] = False

    estimator = NadarayaWatsonEstimator(kernel="gaussian", is_percentile=False)
    estimated_value = estimator.impute(
        row=0,
        column=TARGET_COLUMN,
        data_array=DATA,
        mask_array=mask,
        distance_threshold=1.0,
        data_type=Scalar(),
    )
    assert np.isnan(estimated_value)


def test_returns_nan_when_no_row_is_within_the_bandwidth() -> None:
    """A box kernel narrower than every distance leaves no contributing row."""
    data = np.array([[0.0, 0.0], [5.0, 5.0]])
    mask = np.array([[True, False], [True, True]])

    estimator = NadarayaWatsonEstimator(kernel="box", is_percentile=False)
    estimated_value = estimator.impute(
        row=0,
        column=1,
        data_array=data,
        mask_array=mask,
        distance_threshold=1.0,
        data_type=Scalar(),
    )
    assert np.isnan(estimated_value)


def test_tuple_threshold_is_rejected() -> None:
    """The estimator takes a single bandwidth, not a row/column pair."""
    estimator = NadarayaWatsonEstimator()
    with pytest.raises(ValueError, match="single distance threshold"):
        estimator.impute(
            row=0,
            column=TARGET_COLUMN,
            data_array=DATA,
            mask_array=MASK,
            distance_threshold=(1.0, 1.0),
            data_type=Scalar(),
        )


def test_non_scalar_data_type_is_rejected() -> None:
    """A kernel-weighted average is only defined for scalar outcomes."""
    from nsquared.nnimputer import DataType

    class _NotScalar(DataType):
        def distance(self, obj1: float, obj2: float) -> float:
            return 0.0

        def average(self, object_list: np.ndarray) -> float:
            return 0.0

    estimator = NadarayaWatsonEstimator()
    with pytest.raises(ValueError, match="only defined for scalar outcomes"):
        estimator.impute(
            row=0,
            column=TARGET_COLUMN,
            data_array=DATA,
            mask_array=MASK,
            distance_threshold=1.0,
            data_type=_NotScalar(),
        )


def test_works_through_nearest_neighbor_imputer() -> None:
    """The estimator composes with NearestNeighborImputer like the others."""
    imputer = NearestNeighborImputer(
        NadarayaWatsonEstimator(kernel="gaussian", is_percentile=False),
        Scalar(),
        distance_threshold=1.0,
    )
    estimated_value = imputer.impute(
        row=0,
        column=TARGET_COLUMN,
        data_array=DATA,
        mask_array=MASK,
    )

    weights = np.exp(-0.5 * EXPECTED_ROW_DISTS / 1.0**2)
    expected = weights @ DATA[:, TARGET_COLUMN] / weights.sum()
    assert np.isclose(estimated_value, expected)


def test_nan_at_masked_positions_is_ignored() -> None:
    """Values stored at masked-out positions must not affect the estimate.

    Regression test: the weighted sum contracted over every row, so a nan in
    the target column at an unobserved row (weight zero) still produced nan.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((12, 8))
    mask = (rng.random((12, 8)) < 0.7).astype(int)
    with_nan = np.where(mask == 1, data, np.nan)
    with_zero = np.where(mask == 1, data, 0.0)

    r, c = map(int, np.argwhere(mask == 0)[0])
    imputer = NearestNeighborImputer(
        NadarayaWatsonEstimator(), Scalar(), distance_threshold=0.5
    )
    a = imputer.impute(r, c, with_nan, mask)
    b = imputer.impute(r, c, with_zero, mask)
    assert np.isfinite(a)
    assert np.isclose(a, b)
