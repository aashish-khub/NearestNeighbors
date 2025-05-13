"""Tests for the vanilla NN."""

import numpy as np

from nearest_neighbors.vanilla_nn import row_row, col_col

# initialize the number of rows and columns
ROWS = 10
COLS = 10
# initialize the NN imputer
imputer = row_row(
    distance_threshold=1,
    is_percentile=False,
)


def test_constant_imputation_0() -> None:
    """Sanity Check: Given a matrix with constant values,
    the imputation should be the same constant value.
    """
    data = np.zeros((ROWS, COLS), dtype=float)
    mask = np.ones((ROWS, COLS), dtype=bool)

    for r, c in np.ndindex(data.shape):
        estimated_value = imputer.impute(
            row=r,
            column=c,
            data_array=data,
            mask_array=mask,
        )
        assert estimated_value == 0


def test_constant_imputation_0_5() -> None:
    """Sanity Check: Given a matrix with constant values,
    the imputation should be the same constant value.
    """
    data = np.ones((ROWS, COLS), dtype=float) * 0.5
    mask = np.ones((ROWS, COLS), dtype=bool)

    for r, c in np.ndindex(data.shape):
        estimated_value = imputer.impute(
            row=r,
            column=c,
            data_array=data,
            mask_array=mask,
        )
        assert estimated_value == 0.5


def test_constant_imputation_1() -> None:
    """Sanity Check: Given a matrix with constant values,
    the imputation should be the same constant value.
    """
    data = np.ones((ROWS, COLS), dtype=float)
    mask = np.ones((ROWS, COLS), dtype=bool)

    for r, c in np.ndindex(data.shape):
        estimated_value = imputer.impute(
            row=r,
            column=c,
            data_array=data,
            mask_array=mask,
        )
        assert estimated_value == 1


def test_half_ones_half_zeros_observed_1() -> None:
    """Given a data matrix with the first half of the columns being ones
    and the second half of the columns being zeros, but only the first
    half is observed, the imputed value should be one for the first half
    and zero for the second half.
    """
    data = np.ones((ROWS, COLS), dtype=float)
    data[:, COLS // 2 :] = 0
    mask = np.ones((ROWS, COLS), dtype=bool)
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
                assert estimated_value == 1
            else:
                estimated_value = imputer.impute(
                    row=r,
                    column=c,
                    data_array=data,
                    mask_array=mask,
                )
                assert np.isnan(estimated_value)


def test_half_ones_half_zeros_observed_2() -> None:
    """Given a data matrix with the first half of the columns being ones
    and the second half of the columns being zeros, but only the second
    half is observed, the imputed value should be one for the first half
    and zero for the second half.
    """
    data = np.ones((ROWS, COLS), dtype=float)
    data[:, COLS // 2 :] = 0
    mask = np.ones((ROWS, COLS), dtype=bool)
    mask[:, : COLS // 2] = 0

    for r in range(ROWS):
        for c in range(COLS):
            if c < COLS // 2:
                estimated_value = imputer.impute(
                    row=r,
                    column=c,
                    data_array=data,
                    mask_array=mask,
                )
                assert np.isnan(estimated_value)
            else:
                estimated_value = imputer.impute(
                    row=r,
                    column=c,
                    data_array=data,
                    mask_array=mask,
                )
                assert estimated_value == 0


def test_example_1() -> None:
    """Test the entry at (1, 2)."""
    DATA = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 1],
        ],
        dtype=float,
    )
    MASK = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    imputer = row_row(
        distance_threshold=0,
    )
    estimated_value = imputer.impute(
        row=1,
        column=2,
        data_array=DATA,
        mask_array=MASK,
    )
    assert np.isclose(estimated_value, 0.5)


def test_example_2() -> None:
    """Test the entry at (1, 2) using row-row imputation."""
    DATA = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 1],
        ],
        dtype=float,
    )
    MASK = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    imputer = row_row(
        distance_threshold=0,
    )
    estimated_value = imputer.impute(
        row=1,
        column=2,
        data_array=DATA,
        mask_array=MASK,
    )
    assert np.isclose(estimated_value, 2 / 3)


def test_example_3() -> None:
    """Test the entry at (1, 2) using col-col imputation.

    NOTE: we simply transpose the DATA and MASK matrices
    from test_example_2.
    """
    DATA = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )
    MASK = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    imputer = col_col(
        distance_threshold=0,
    )
    estimated_value = imputer.impute(
        row=2,
        column=1,
        data_array=DATA,
        mask_array=MASK,
    )
    assert np.isclose(estimated_value, 2 / 3)
