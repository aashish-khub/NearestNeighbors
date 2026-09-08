"""k-nearest-neighbor imputation, implemented directly on NumPy.

Matches the semantics of ``sklearn.impute.KNNImputer``: each missing entry is
filled with the average of that column's value among the ``k`` rows closest to
it under the nan-aware Euclidean metric, restricted to rows that actually
observe the column being filled.

Implemented here rather than depending on scikit-learn so that the baselines
stay dependency-free; see ``tests/test_knn_baseline.py`` for the equivalence
check against scikit-learn.
"""

import numpy as np
import numpy.typing as npt


def _nan_euclidean_distances(matrix: npt.NDArray) -> npt.NDArray:
    """Pairwise nan-aware Euclidean distances between the rows of ``matrix``.

    Coordinates where either row is missing are skipped, and the running sum is
    scaled by ``n_features / n_present`` so that rows overlapping on few
    coordinates are not spuriously close. This is the metric scikit-learn calls
    ``nan_euclidean``.

    Args:
        matrix (npt.NDArray): Data with ``np.nan`` in the missing positions.

    Returns:
        npt.NDArray: Square distance matrix. Pairs with no overlapping observed
            coordinate get ``np.inf``.

    """
    observed = ~np.isnan(matrix)
    filled = np.where(observed, matrix, 0.0)

    # Squared differences, counting only coordinates observed in both rows.
    both = observed.astype(float) @ observed.astype(float).T
    squares = filled**2
    cross = (squares * observed) @ observed.T.astype(float)
    total = cross + cross.T - 2.0 * (filled @ filled.T)

    n_features = matrix.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = total * (n_features / both)
    scaled[both == 0] = np.inf
    # Floating point can leave tiny negatives on the diagonal.
    return np.sqrt(np.maximum(scaled, 0.0))


def knn_impute(
    X: npt.NDArray, n_neighbors: int = 5, weights: str = "uniform"
) -> npt.NDArray:
    """Impute missing entries from the k nearest rows.

    Args:
        X (npt.NDArray): N x T matrix with missing values as ``np.nan``.
        n_neighbors (int): Number of donor rows to average over. Fewer are used
            when not enough rows observe the target column.
        weights (str): ``"uniform"`` to average donors equally, or
            ``"distance"`` to weight them by the inverse of their distance.

    Raises:
        ValueError: If ``X`` is not two-dimensional, if ``n_neighbors`` is not
            positive, or if ``weights`` is not a supported scheme.

    Returns:
        npt.NDArray: The imputed matrix. Observed entries are returned
            unchanged. Columns with no observed value anywhere stay ``np.nan``.

    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-dimensional matrix, got shape {X.shape}")
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be at least 1, got {n_neighbors}")
    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")

    missing = np.isnan(X)
    if not missing.any():
        return X.copy()

    distances = _nan_euclidean_distances(X)
    np.fill_diagonal(distances, np.inf)  # a row is not its own neighbor
    observed = ~missing
    imputed = X.copy()

    for row, col in zip(*np.nonzero(missing)):
        # Only rows that observe this column can donate a value.
        donors = np.nonzero(observed[:, col] & np.isfinite(distances[row]))[0]
        if donors.size == 0:
            continue

        nearest = donors[np.argsort(distances[row, donors], kind="stable")][
            :n_neighbors
        ]
        values = X[nearest, col]

        if weights == "uniform":
            imputed[row, col] = values.mean()
        else:
            donor_distances = distances[row, nearest]
            if np.any(donor_distances == 0):
                # Coincident rows dominate; sklearn averages just those.
                imputed[row, col] = values[donor_distances == 0].mean()
            else:
                inverse = 1.0 / donor_distances
                imputed[row, col] = float(np.sum(values * inverse) / np.sum(inverse))

    return imputed


def knn_impute_columnwise(
    X: npt.NDArray, n_neighbors: int = 5, weights: str = "uniform"
) -> npt.NDArray:
    """Impute by averaging over similar *columns* rather than similar rows.

    Matrix completion has no privileged orientation, so the column-wise variant
    is provided as a second baseline. Equivalent to transposing, calling
    :func:`knn_impute`, and transposing back.

    Args:
        X (npt.NDArray): N x T matrix with missing values as ``np.nan``.
        n_neighbors (int): Number of donor columns to average over.
        weights (str): ``"uniform"`` or ``"distance"``.

    Returns:
        npt.NDArray: The imputed matrix.

    """
    return knn_impute(np.asarray(X, dtype=float).T, n_neighbors, weights).T
