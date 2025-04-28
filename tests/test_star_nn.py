"""Tests for the Star NN imputer."""

import numpy as np
import pytest
from nearest_neighbors.star_nn import star_nn
from nearest_neighbors.data_types import Scalar

# Initialize constants
ROWS = 4
COLS = 4

def test_constant_imputation() -> None:
    """Test imputation on a matrix with constant values.
    The imputed value should be the same constant value.
    """
    data = np.ones((ROWS, COLS))
    mask = np.ones((ROWS, COLS))
    mask[1, 1] = 0  # Make one entry missing
    
    imputer = star_nn(noise_variance=0.1, delta=0.05)
    imputer.set_row_distances(data, mask)
    
    imputed_value = imputer.impute(1, 1, data, mask)
    assert np.isclose(imputed_value, 1.0)

def test_no_observed_values() -> None:
    """Test imputation when there are no observed values in the column.
    Should return NaN.
    """
    data = np.ones((ROWS, COLS))
    mask = np.ones((ROWS, COLS))
    mask[:, 1] = 0  # Make entire column missing
    
    imputer = star_nn(noise_variance=0.1, delta=0.05)
    imputer.set_row_distances(data, mask)
    
    imputed_value = imputer.impute(1, 1, data, mask)
    assert np.isnan(imputed_value)

def test_weight_calculation() -> None:
    """Test that weights sum to approximately 1 and are non-negative."""
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 0.0, 3.1, 4.1],  # Second value will be imputed
        [1.2, 2.2, 3.2, 4.2],
        [1.3, 2.3, 3.3, 4.3]
    ])
    mask = np.ones((ROWS, COLS))
    mask[1, 1] = 0
    
    imputer = star_nn(noise_variance=0.1, delta=0.05)
    imputer.set_row_distances(data, mask)
    
    # Get the weights through imputation
    imputed_value = imputer.impute(1, 1, data, mask)
    
    # The imputed value should be a weighted average of the observed values
    # in the same column, so it should be between min and max of those values
    observed_values = data[mask[:, 1] == 1, 1]
    assert observed_values.min() <= imputed_value <= observed_values.max()

def test_noise_variance_estimation() -> None:
    """Test that noise variance estimation converges."""
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
        [1.2, 2.2, 3.2, 4.2],
        [1.3, 2.3, 3.3, 4.3]
    ])
    mask = np.ones((ROWS, COLS))
    mask[1, 1] = 0
    
    imputer = star_nn(delta=0.05)  # Don't set noise_variance initially
    imputed_data = imputer.fit(data, mask, max_iterations=10)
    
    # Check that noise variance was estimated and is positive
    assert imputer.noise_variance is not None
    assert imputer.noise_variance > 0

def test_row_distances_calculation() -> None:
    """Test that row distances are properly calculated."""
    data = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0]
    ])
    mask = np.ones((3, 3))
    
    imputer = star_nn(noise_variance=0.1, delta=0.05)
    imputer.set_row_distances(data, mask)
    
    # Check that distances are symmetric
    assert np.allclose(imputer.row_distances, imputer.row_distances.T)
    # Check diagonal is zero
    assert np.allclose(np.diag(imputer.row_distances), 0)
    # Check distances are non-negative
    assert np.all(imputer.row_distances >= 0)

def test_convergence() -> None:
    """Test that the fitting process converges within max_iterations."""
    data = np.random.randn(ROWS, COLS)
    mask = np.ones((ROWS, COLS))
    mask[1, 1] = 0
    
    imputer = star_nn(noise_variance=0.1, delta=0.05)
    max_iterations = 5
    imputed_data = imputer.fit(data, mask, max_iterations=max_iterations)
    
    # Check that imputed data has the same shape as input
    assert imputed_data.shape == data.shape
    # Check that non-missing values are unchanged
    assert np.allclose(imputed_data[mask == 1], data[mask == 1])

def test_delta_parameter() -> None:
    """Test that different delta values affect the imputation."""
    data = np.random.randn(ROWS, COLS)
    mask = np.ones((ROWS, COLS))
    mask[1, 1] = 0
    
    # Compare imputation with different delta values
    imputer1 = star_nn(noise_variance=0.1, delta=0.01)
    imputer2 = star_nn(noise_variance=0.1, delta=0.5)
    
    imputer1.set_row_distances(data, mask)
    imputer2.set_row_distances(data, mask)
    
    value1 = imputer1.impute(1, 1, data, mask)
    value2 = imputer2.impute(1, 1, data, mask)
    
    # Values should be different for very different deltas
    assert not np.isclose(value1, value2)

def test_edge_case_single_observation() -> None:
    """Test imputation when there's only one observed value in the column."""
    data = np.ones((ROWS, COLS))
    mask = np.zeros((ROWS, COLS))
    mask[0, 1] = 1  # Only one observation in column 1
    
    imputer = star_nn(noise_variance=0.1, delta=0.05)
    imputer.set_row_distances(data, mask)
    
    imputed_value = imputer.impute(1, 1, data, mask)
    # Should return the only observed value
    assert np.isclose(imputed_value, data[0, 1])

def test_invalid_inputs() -> None:
    """Test that invalid inputs raise appropriate errors."""
    imputer = star_nn(noise_variance=0.1, delta=0.05)
    
    # Test with mismatched dimensions
    with pytest.raises(ValueError):
        data = np.ones((3, 3))
        mask = np.ones((4, 4))
        imputer.set_row_distances(data, mask)
    
    # Test with invalid noise variance
    with pytest.raises(ValueError):
        imputer = star_nn(noise_variance=-1, delta=0.05)
    
    # Test with invalid delta
    with pytest.raises(ValueError):
        imputer = star_nn(noise_variance=0.1, delta=1.5) 