"""Tests for the Nadaraya-Watson estimator."""

import numpy as np

from nearest_neighbors.nadaraya_watson import NadarayaWatsonEstimator
from nearest_neighbors.data_types import Scalar
from nearest_neighbors.nnimputer import NearestNeighborImputer

# initialize the number of rows and columns
ROWS = 10
COLS = 10
# initialize the NN imputer
estimator = NadarayaWatsonEstimator(kernel="gaussian")
data_type = Scalar()
imputer = NearestNeighborImputer(estimator, data_type, distance_threshold=1)


def test_constant_imputation_0() -> None:
    """Sanity Check: Given a matrix with constant values,
    the imputation should be the same constant value.
    """
    data = np.zeros((ROWS, COLS))
    mask = np.ones((ROWS, COLS))

    for r, c in np.ndindex(data.shape):
        estimated_value = imputer.impute(
            row=r,
            column=c,
            data_array=data,
            mask_array=mask,
        )
        assert estimated_value == 0


def test_half_ones_half_zeros_observed_1() -> None:
    """Given a data matrix with the first half of the columns being ones
    and the second half of the columns being zeros, but only the first
    half is observed, the imputed value should be one for the first half
    and zero for the second half.
    """
    data = np.ones((ROWS, COLS))
    data[:, COLS // 2 :] = 0
    mask = np.ones((ROWS, COLS))
    mask[:, COLS // 2 :] = 0

    for r in range(ROWS):
        for c in range(COLS):
            if c < COLS // 2:
                estimated_value = imputer.impute(
                    row=r,
                    column=c,
                    data_array=data,
                    mask_array=mask,
                )
                assert estimated_value == 1, (
                    f"Expected 1 for row {r} and column {c}, but got {estimated_value}"
                )
            else:
                # import pdb; pdb.set_trace()
                estimated_value = imputer.impute(
                    row=r,
                    column=c,
                    data_array=data,
                    mask_array=mask,
                )
                assert np.isnan(estimated_value), (
                    f"Expected NaN for row {r} and column {c}, but got {estimated_value}"
                )
