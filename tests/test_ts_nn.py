import numpy as np
from nearest_neighbors.ts_nn import ts_nn


def test_full_neighborhood() -> None:
    """Test imputation when threshold is high so that all possible neighbors are used.

    For a 4x4 fully observed matrix with a missing cell at (2,2), the imputed value
    should be the average of the cross-product of all other rows and columns.
    """
    X = np.arange(16).reshape(4, 4).astype(float)
    A = np.ones((4, 4))
    # Make (2,2) missing.
    A[2, 2] = 0
    # With very high thresholds, neighbors are all rows (except row 2) and all columns (except col 2).
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=2, column=2, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # Neighbors: rows {0,1,3} and cols {0,1,3}
    vals = [
        X[0, 0],
        X[0, 1],
        X[0, 3],
        X[1, 0],
        X[1, 1],
        X[1, 3],
        X[3, 0],
        X[3, 1],
        X[3, 3],
    ]
    expected = np.mean(vals)
    assert np.isclose(estimated_value, expected), (
        f"Expected {expected}, got {estimated_value}"
    )


def test_single_neighbor() -> None:
    """Test imputation when there is exactly one neighbor.

    In a 2x2 matrix with one missing value at (1,1), if the only neighbor is row 0 and col 0,
    then the imputed value should equal X[0,0].
    """
    X = np.array([[5, 5], [5, 0]], dtype=float)
    A = np.array([[1, 1], [1, 0]])
    # With high thresholds, row neighbor of row 1 is {0} and column neighbor of col 1 is {0}.
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    expected = X[0, 0]  # 5
    assert np.isclose(estimated_value, expected), (
        f"Expected {expected}, got {estimated_value}"
    )


def test_partial_neighborhood() -> None:
    """Test imputation on a matrix with some found neighbors.

    For a 3x3 matrix with a missing value at (1,1) and high thresholds,
    the imputed value should be the average of the cross-product of rows {0,2} and cols {0,2}.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    A = np.ones_like(X)
    # Make (1,1) missing.
    A[1, 1] = 0
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # Neighbors: rows {0,2}, cols {0,2}
    vals = [X[0, 0], X[0, 2], X[2, 0], X[2, 2]]
    expected = np.mean(vals)
    assert np.isclose(estimated_value, expected), (
        f"Expected {expected}, got {estimated_value}"
    )


def test_no_neighbors_fallback() -> None:
    """Test fallback when no neighbor is found.

    With threshold zero, no neighbor is selected. Then if the target cell is missing,
    the imputation should return np.nan
    """
    X = np.array([[20, 20], [30, 40]])
    A = np.array([[0, 1], [1, 1]])
    imputer = ts_nn(
        distance_threshold_row=0, distance_threshold_col=0, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=0, column=0, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    assert np.isnan(estimated_value), f"Expected NaN, got {estimated_value}"


def test_multiple_missing_in_neighborhood() -> None:
    """Test imputation when some neighbor entries are also missing.

    In a 3x3 matrix with target missing at (1,1) and an additional missing value at (0,0)
    in the neighborhood, the average is computed only over the observed values.
    """
    X = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
    A = np.ones_like(X)
    # Make target (1,1) missing and also (0,0) missing.
    A[1, 1] = 0
    A[0, 0] = 0
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # Neighbors: rows {0,2}, cols {0,2} → cross product = {(0,0), (0,2), (2,0), (2,2)}
    # (0,0) is missing, so use (0,2), (2,0), (2,2)
    vals = [X[0, 2], X[2, 0], X[2, 2]]
    expected = np.mean(vals)
    assert np.isclose(estimated_value, expected), (
        f"Expected {expected}, got {estimated_value}"
    )


def test_imputation_stability() -> None:
    """Test that repeated imputation on the same input returns the same value.

    This ensures that the TSNN imputation is deterministic.
    """
    X = np.random.rand(5, 5)
    A = np.ones((5, 5))
    # Make target (2,2) missing.
    A[2, 2] = 0
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    val1 = imputer.impute(
        row=2, column=2, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    val2 = imputer.impute(
        row=2, column=2, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    print(val1)
    assert np.isclose(val1, val2), f"Imputation not stable: {val1} vs {val2}"


def test_random_noise_imputation() -> None:
    """Test that imputation on a random matrix with missing entries does not return NaN.

    This checks that the TSNN imputer can handle randomness and produce a finite result.
    """
    np.random.seed(0)
    X = np.random.rand(6, 6)
    A = np.ones((6, 6))
    # Introduce missingness at a few random locations.
    A[1, 3] = 0
    A[4, 2] = 0
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=3, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    assert np.isfinite(estimated_value), (
        f"Imputed value is not finite: {estimated_value}"
    )


def test_edge_case_empty_matrix() -> None:
    """Test imputation on an empty matrix (no rows or columns)."""
    X = np.empty((0, 0))
    A = np.empty((0, 0))
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    try:
        imputer.impute(
            row=0, column=0, data_array=X, mask_array=A, allow_self_neighbor=False
        )
    except IndexError:
        pass  # Expect an IndexError due to the empty matrix
    else:
        raise AssertionError("Expected IndexError for empty matrix.")


def test_single_row_missing() -> None:
    """Test imputation with a matrix where an entire row is missing."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    A = np.ones_like(X)
    # Make row 1 entirely missing.
    A[1, :] = 0
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # nan because N_row is the empty set, and the cross product of an empty set with any set is empty.
    assert np.isnan(estimated_value), f"Expected NaN, got {estimated_value}"


def test_single_column_missing() -> None:
    """Test imputation with a matrix where an entire column is missing."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    A = np.ones_like(X)
    # Make column 1 entirely missing.
    A[:, 1] = 0
    imputer = ts_nn(
        distance_threshold_row=1e9, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # nan because N_col is the empty set, and the cross product of any set with an empty set is empty.
    assert np.isnan(estimated_value), f"Expected NaN, got {estimated_value}"


def test_multiple_neighbors_with_finite_thresholds() -> None:
    """Test imputation with multiple neighbors but finite distance thresholds."""
    X = np.array([[1, 2, 30], [4, 5, 6], [7, 8, 9]], dtype=float)
    A = np.ones_like(X)
    # Make (1,1) missing, and rows 0 and 2 should be considered as neighbors.
    A[1, 1] = 0
    imputer = ts_nn(
        distance_threshold_row=1000, distance_threshold_col=10000, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # Neighbors: rows {0, 2}, cols {0, 2}
    vals = [X[0, 0], X[0, 2], X[2, 0], X[2, 2]]
    expected = np.mean(vals)
    assert np.isclose(estimated_value, expected), (
        f"Expected {expected}, got {estimated_value}"
    )


def test_find_one_neighbor() -> None:
    """Test imputation with a singular near-infinite neighbor but finite distance thresholds."""
    X = np.array([[1, 2, 30], [4, 5, 6], [7, 8, 9e99]], dtype=float)
    A = np.ones_like(X)
    A[1, 1] = 0
    imputer = ts_nn(
        distance_threshold_row=1000, distance_threshold_col=1000, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # Neighbors: rows {0}, cols {0}
    vals = [X[0, 0]]
    expected = np.mean(vals)
    assert np.isclose(estimated_value, expected), (
        f"Expected {expected}, got {estimated_value}"
    )


def test_low_row_high_col_threshold() -> None:
    """Test imputation with a low row threshold and a high column threshold.

    In this case, only row 0 is selected as the neighbor, but all columns {0, 1, 2}
    are considered as neighbors due to the high column threshold.
    """
    X = np.array([[1, 2, 3], [10, 20, 30], [10000, 20000, 30000]], dtype=float)
    A = np.ones_like(X)
    # Make (1,1) missing.
    A[1, 1] = 0
    imputer = ts_nn(
        distance_threshold_row=1000, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # Neighbors: row {0}, cols {0, 1, 2} → cross product = {(0,0), (0,1), (0,2)}
    vals = [X[0, 0], X[0, 1], X[0, 2]]
    expected = np.mean(vals)
    assert np.isclose(estimated_value, expected), (
        f"Expected {expected}, got {estimated_value}"
    )


def test_low_col_high_row_threshold() -> None:
    """Test imputation with a low column threshold and a high row threshold.

    In this case, only column 0 is selected as the neighbor, but all rows {0, 1, 2}
    are considered as neighbors due to the high column threshold.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 20000, 30000]], dtype=float)
    A = np.ones_like(X)
    # Make (1,1) missing.
    A[1, 1] = 0
    imputer = ts_nn(
        distance_threshold_row=1000, distance_threshold_col=1e9, is_percentile=False
    )
    estimated_value = imputer.impute(
        row=1, column=1, data_array=X, mask_array=A, allow_self_neighbor=False
    )
    # Neighbors: row {0}, cols {0, 1, 2} → cross product = {(0,0), (0,1), (0,2)}
    vals = [X[0, 0], X[0, 1], X[0, 2]]
    expected = np.mean(vals)
    assert np.isclose(estimated_value, expected), (
        f"Expected {expected}, got {estimated_value}"
    )
