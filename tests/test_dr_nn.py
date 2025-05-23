import numpy as np
from nsquared.dr_nn import dr_nn
import pytest

ROWS = 10
COLS = 10


def test_drnn_all0() -> None:
    """Test the drnn imputer with all values 0."""
    imputer = dr_nn(distance_threshold_row=1, distance_threshold_col=1)
    data = np.zeros([ROWS, COLS])
    mask = np.ones([ROWS, COLS], dtype=bool)

    imputed_value = imputer.impute(
        row=0,
        column=0,
        data_array=data,
        mask_array=mask,
    )

    # imputed value should be 0
    assert imputed_value == 0


def test_drnn_all1() -> None:
    """Test the drnn imputer with all values 1."""
    imputer = dr_nn(distance_threshold_row=1, distance_threshold_col=1)
    data = np.ones([ROWS, COLS])
    mask = np.ones([ROWS, COLS], dtype=bool)

    imputed_value = imputer.impute(
        row=0,
        column=0,
        data_array=data,
        mask_array=mask,
    )

    # imputed value should be 1
    assert imputed_value == 1


def test_drnn_avg1() -> None:
    """Test the drnn imputer with single doubly robust estimate to average over."""
    imputer = dr_nn(distance_threshold_row=1, distance_threshold_col=1)
    data = np.array([[np.nan, 1, 4.0], [1, 1, 3.0], [1, 5, 2]])

    mask = np.array(
        [
            [0, 1, 0],  # Missing value at [0, 0]
            [1, 1, 0],
            [0, 0, 0],
        ]
    )

    # Manually call the imputation logic for the missing value at [0, 0]
    imputed_value = imputer.impute(
        row=0,
        column=0,
        data_array=data,
        mask_array=mask,
    )

    # The imputed value should be 1 + 1 - 1 = 1
    assert imputed_value == 1


def test_drnn_avg2() -> None:
    """Test the drnn imputer with multiple doubly robust estimates to average over."""
    imputer = dr_nn(distance_threshold_row=0.5, distance_threshold_col=0.5)
    data = np.array([[0, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [1, 1, 0, 1]])
    # neighbors are row 1, row 3, col 1, col 3. So estimate is:
    # ((1 + 1 - 1) + (1 + 1 - 1) + (1 + 1 - 1) + (1 + 1 - 1)) / 4 = 1

    mask = np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    imputed_value = imputer.impute(
        row=0,
        column=0,
        data_array=data,
        mask_array=mask,
    )
    assert imputed_value == 1


if __name__ == "__main__":
    pytest.main()
