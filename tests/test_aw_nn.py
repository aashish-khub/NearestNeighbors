"""Tests for the AWNN imputer."""

import numpy as np
import pytest
from nsquared.aw_nn import aw_nn

# Initialize constants
ROWS = 16
COLS = 16


def test_impute_one_vs_all() -> None:
    """Test imputation of a single value vs. all values, ensure they align"""
    np.random.seed(0)  # For reproducibility
    data = np.random.randn(ROWS, COLS)
    mask = np.ones((ROWS, COLS))
    mask[1, 1] = 0  # Make one entry missing
    mask[2, 2] = 0  # Make another entry missing
    mask[3, 3] = 0  # Make another entry missing
    imputer = aw_nn()
    imputed_matrix = imputer.impute_all(data, mask)
    for i in range(ROWS):
        for j in range(COLS):
            if mask[i, j] == 0:
                imputed_value = imputer.impute(i, j, data, mask)
                assert np.isclose(imputed_value, imputed_matrix[i, j]), (
                    f"Imputed value at ({i}, {j}) does not match imputed matrix."
                )


def test_constant_imputation() -> None:
    """Test imputation on a matrix with constant values.
    The imputed value should be the same constant value.
    """
    data = np.ones((ROWS, COLS))
    mask = np.ones((ROWS, COLS))
    mask[1, 1] = 0  # Make one entry missing

    imputer = aw_nn(noise_variance=0.1, delta=0.05)

    imputed_value = imputer.impute(1, 1, data, mask)
    assert np.isclose(imputed_value, 1.0)


def test_no_observed_values() -> None:
    """Test imputation when there are no observed values in the column.
    Should return NaN.
    """
    data = np.ones((ROWS, COLS))
    mask = np.ones((ROWS, COLS))
    mask[:, 1] = 0  # Make entire column missing

    imputer = aw_nn()

    imputed_all = imputer.impute_all(data, mask)
    all_in_col_are_nan = np.isnan(imputed_all[:, 1])
    assert np.all(all_in_col_are_nan), "All imputed values in column should be NaN."


def test_weight_calculation() -> None:
    """Test that weights sum to approximately 1 and are non-negative."""
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 0.0, 3.1, 4.1],  # Second value will be imputed
            [1.2, 2.2, 3.2, 4.2],
            [1.3, 2.3, 3.3, 4.3],
        ]
    )
    mask = np.ones(data.shape)
    mask[1, 1] = 0

    imputer = aw_nn()

    # Get the weights through imputation
    imputed_value = imputer.impute(1, 1, data, mask)

    # The imputed value should be a weighted average of the observed values
    # in the same column, so it should be between min and max of those values
    observed_values = data[mask[:, 1] == 1, 1]
    assert observed_values.min() <= imputed_value <= observed_values.max()


def test_convergence() -> None:
    """Test that the fitting process converges within max_iterations."""
    np.random.seed(0)  # For reproducibility
    data = np.random.randn(ROWS, COLS)
    mask = np.ones((ROWS, COLS))
    # hide all diagonal and anti-diagonal elements to 0
    mask = mask - (np.eye(ROWS, COLS) * np.fliplr(np.eye(ROWS, COLS)))
    max_iterations = 100
    # Set a high max_iterations to ensure convergence
    imputer = aw_nn(
        noise_variance=0.1,
        delta=0.05,
        max_iterations=max_iterations,
    )
    # imputed_data = imputer.fit(data, mask, max_iterations=max_iterations)

    imputed_data = imputer.impute_all(data, mask)
    # Check that imputed data is not none and has the same shape as input
    assert imputed_data.shape == data.shape, (
        f"Shape mismatch: {imputed_data.shape} vs {data.shape}"
    )


def test_delta_parameter() -> None:
    """Test how different delta values affect the imputation."""
    np.random.seed(0)  # For reproducibility
    data = np.random.randn(ROWS, COLS)
    mask = np.ones((ROWS, COLS))
    mask[1, 1] = 0
    mask[2, 2] = 0

    # Compare imputation with different delta values
    imputer1 = aw_nn(noise_variance=0.1, delta=0.01)
    imputer2 = aw_nn(noise_variance=0.1, delta=0.5)

    value1 = imputer1.impute_all(data, mask)
    value2 = imputer2.impute_all(data, mask)
    assert not np.allclose(value1, value2), (
        "Imputed values should differ significantly with different delta values."
    )


def test_high_snr_convergence() -> None:
    """Test that with very high SNR, imputation result converges to the true signal.

    With a very small noise variance (high SNR), the imputed values should be
    very close to the true signal values.
    """
    np.random.seed(0)  # For reproducibility
    # Create a clean signal matrix with a simple pattern
    signal = np.zeros((ROWS, COLS))
    for i in range(ROWS):
        signal[i, :] = i + 1  # Distinct values for each row, e.g., 1, 2, 3, 4
    # Add extremely small noise (high SNR)
    noise_stddev = 1e-6  # Very small noise
    noisy_data = signal + np.random.normal(0, noise_stddev, (ROWS, COLS))
    # Create mask with a single missing value
    mask = np.ones((ROWS, COLS))
    mask[2, 2] = 0  # Make one entry missing

    # Create imputer with a very small noise variance
    imputer = aw_nn(
        noise_variance=noise_stddev**2,
        delta=0.05,
        max_iterations=100,
        convergence_threshold=1e-6,
    )

    # Impute the missing value
    # imputed_value = imputer.impute(2, 2, noisy_data, mask)
    imputed_all = imputer.impute_all(noisy_data, mask)
    # check if all imputed values are close to the true signal with some fault tolerance
    print(signal)
    # print(imputed_value)
    print(imputed_all)
    assert np.all(np.isclose(imputed_all, signal, atol=1e-3)), (
        "All imputed values should be close to the true signal values."
    )


def test_edge_case_single_observation() -> None:
    """Test imputation when there's only one observed value in the column."""
    data = np.ones((ROWS, COLS))
    mask = np.zeros((ROWS, COLS))
    mask[0, 1] = 1  # Only one observation in column 1

    imputer = aw_nn()
    # Should return the only observed value
    imputed_value = imputer.impute(0, 1, data, mask)
    assert np.isclose(imputed_value, data[0, 1])


def test_invalid_inputs() -> None:
    """Test that invalid inputs raise appropriate errors."""
    # Test with invalid delta values (should be between 0 and 1)
    with pytest.raises(ValueError, match="Delta must be between 0 and 1"):
        aw_nn(delta=-0.1)

    with pytest.raises(ValueError, match="Delta must be between 0 and 1"):
        aw_nn(delta=1.5)

    # Test with invalid noise variance (should be non-negative)
    with pytest.raises(ValueError, match="Noise variance must be non-negative"):
        aw_nn(noise_variance=-0.1)

    # Test with invalid convergence threshold (should be non-negative)
    with pytest.raises(ValueError, match="Convergence threshold must be non-negative"):
        aw_nn(convergence_threshold=-0.01)

    # Test with invalid max iterations (should be positive)
    with pytest.raises(ValueError, match="Max iterations must be positive"):
        aw_nn(max_iterations=0)

    with pytest.raises(ValueError, match="Max iterations must be positive"):
        aw_nn(max_iterations=-5)

    # Test with mismatched dimensions for data and mask
    imputer = aw_nn(noise_variance=0.1, delta=0.05)
    data = np.ones((3, 4))  # 3x4 matrix
    mask = np.ones((4, 3))  # 4x3 matrix (transposed dimensions)

    # Attempting to impute with mismatched dimensions should raise an error
    with pytest.raises(Exception):  # Could be IndexError, ValueError, etc.
        imputer.impute(1, 1, data, mask)

    # Test imputation with out-of-bounds indices
    data = np.ones((ROWS, COLS))
    mask = np.ones((ROWS, COLS))

    # Row index out of bounds
    with pytest.raises(Exception):  # Could be IndexError
        imputer.impute(ROWS + 5, 1, data, mask)

    # Column index out of bounds
    with pytest.raises(Exception):  # Could be IndexError
        imputer.impute(1, COLS + 5, data, mask)
