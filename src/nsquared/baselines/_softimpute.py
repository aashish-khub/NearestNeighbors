"""SoftImpute matrix completion, implemented directly on NumPy.

This is a from-scratch implementation of the algorithms published in:

    Mazumder, R., Hastie, T., & Tibshirani, R. (2010). Spectral regularization
    algorithms for learning large incomplete matrices. Journal of Machine
    Learning Research, 11, 2287-2322.

    Hastie, T., Mazumder, R., Lee, J. D., & Zadeh, R. (2015). Matrix completion
    and low-rank SVD via fast alternating least squares. Journal of Machine
    Learning Research, 16(1), 3367-3402.

It replaces a dependency on ``fancyimpute``, which pulled in 26 packages
(including four convex solvers that SoftImpute never uses) and whose 0.7.0
release calls ``sklearn.utils.check_array(force_all_finite=...)``, an argument
removed in scikit-learn 1.8. Everything here needs only NumPy.

Defaults match ``fancyimpute.SoftImpute(normalizer=fancyimpute.BiScaler())`` so
that results computed with the previous implementation remain reproducible; see
``tests/test_softimpute.py`` for the numerical agreement check.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

# Fraction of the largest singular value used as the shrinkage level when the
# caller does not supply one.
_DEFAULT_SHRINKAGE_DIVISOR = 50.0
# Guards divisions by a vanishing norm or scale.
_EPS = 1e-9


def _soft_threshold_svd(
    matrix: npt.NDArray, shrinkage_value: float, max_rank: Optional[int] = None
) -> npt.NDArray:
    """Apply the singular value soft-thresholding operator to a matrix.

    This is the operator written ``S_lambda`` in Mazumder et al. (2010): take
    the SVD, shrink each singular value towards zero by ``shrinkage_value``,
    clamp at zero, and rebuild the matrix from what survives.

    Args:
        matrix (npt.NDArray): Matrix to threshold, with no missing entries.
        shrinkage_value (float): Amount to subtract from each singular value.
        max_rank (Optional[int]): Cap on the rank of the result. Defaults to no
            cap.

    Returns:
        npt.NDArray: The soft-thresholded matrix.

    """
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    shrunk = np.maximum(singular_values - shrinkage_value, 0.0)

    rank = int(np.count_nonzero(shrunk))
    if max_rank is not None:
        rank = min(rank, max_rank)
    if rank == 0:
        return np.zeros_like(matrix)

    return (left[:, :rank] * shrunk[:rank]) @ right[:rank]


def _biscale(
    matrix: npt.NDArray,
    observed: npt.NDArray,
    center_rows: bool = True,
    center_columns: bool = True,
    scale_rows: bool = True,
    scale_columns: bool = True,
    max_iters: int = 100,
    tolerance: float = 0.001,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Center and scale rows and columns simultaneously, ignoring missing data.

    Fits the model ``X[i, j] ~ row_center[i] + col_center[j] +
    row_scale[i] * col_scale[j] * Z[i, j]`` by alternating closed-form updates,
    as described in Section 8 of Hastie et al. (2015). Centering and scaling
    interact -- rescaling changes the means -- so the two are iterated to a
    fixed point rather than applied once.

    Args:
        matrix (npt.NDArray): Data with ``np.nan`` in the missing positions.
        observed (npt.NDArray): Boolean mask, ``True`` where data is observed.
        center_rows (bool): Whether to fit row centers.
        center_columns (bool): Whether to fit column centers.
        scale_rows (bool): Whether to fit row scales.
        scale_columns (bool): Whether to fit column scales.
        max_iters (int): Maximum alternating sweeps.
        tolerance (float): Stop once the largest parameter change falls below
            this.

    Returns:
        Tuple: ``(normalized, row_center, col_center, row_scale, col_scale)``.
            ``normalized`` keeps ``np.nan`` in the missing positions.

    """
    n_rows, n_cols = matrix.shape
    row_center = np.zeros(n_rows)
    col_center = np.zeros(n_cols)
    row_scale = np.ones(n_rows)
    col_scale = np.ones(n_cols)

    # Work with zeros in the missing slots and rely on the mask, so that sums
    # below never have to special-case nan.
    filled = np.where(observed, np.nan_to_num(matrix), 0.0)
    row_counts = observed.sum(axis=1)
    col_counts = observed.sum(axis=0)

    for _ in range(max_iters):
        previous = (
            row_center.copy(),
            col_center.copy(),
            row_scale.copy(),
            col_scale.copy(),
        )

        # --- centers, given the current scales ---------------------------
        # Setting the row mean of the normalized residual to zero gives a
        # closed form for row_center, and symmetrically for col_center.
        inverse_scale = 1.0 / (np.outer(row_scale, col_scale) + _EPS)
        if center_rows:
            residual = (filled - col_center[None, :]) * inverse_scale * observed
            weight = inverse_scale * observed
            row_center = np.where(
                row_counts > 0, residual.sum(axis=1) / (weight.sum(axis=1) + _EPS), 0.0
            )
        if center_columns:
            residual = (filled - row_center[:, None]) * inverse_scale * observed
            weight = inverse_scale * observed
            col_center = np.where(
                col_counts > 0, residual.sum(axis=0) / (weight.sum(axis=0) + _EPS), 0.0
            )

        centered = (filled - row_center[:, None] - col_center[None, :]) * observed

        # --- scales, given the current centers ----------------------------
        # Chosen so each row (column) of the normalized residual has unit
        # mean square.
        if scale_rows:
            scaled = centered / (col_scale[None, :] + _EPS)
            mean_square = np.where(
                row_counts > 0,
                (scaled**2 * observed).sum(axis=1) / np.maximum(row_counts, 1),
                1.0,
            )
            row_scale = np.sqrt(np.maximum(mean_square, _EPS))
        if scale_columns:
            scaled = centered / (row_scale[:, None] + _EPS)
            mean_square = np.where(
                col_counts > 0,
                (scaled**2 * observed).sum(axis=0) / np.maximum(col_counts, 1),
                1.0,
            )
            col_scale = np.sqrt(np.maximum(mean_square, _EPS))

        shift = max(
            np.abs(row_center - previous[0]).max(initial=0.0),
            np.abs(col_center - previous[1]).max(initial=0.0),
            np.abs(row_scale - previous[2]).max(initial=0.0),
            np.abs(col_scale - previous[3]).max(initial=0.0),
        )
        if shift < tolerance:
            break

    normalized = (matrix - row_center[:, None] - col_center[None, :]) / (
        np.outer(row_scale, col_scale) + _EPS
    )
    return normalized, row_center, col_center, row_scale, col_scale


def _converged(
    old: npt.NDArray, new: npt.NDArray, missing: npt.NDArray, threshold: float
) -> bool:
    """Test whether the estimates of the missing entries have stopped moving.

    Args:
        old (npt.NDArray): Previous iterate.
        new (npt.NDArray): Current iterate.
        missing (npt.NDArray): Boolean mask, ``True`` where data is missing.
        threshold (float): Relative-change threshold.

    Returns:
        bool: True once the relative change falls below ``threshold``.

    """
    old_missing = old[missing]
    new_missing = new[missing]
    delta = np.sqrt(np.sum((old_missing - new_missing) ** 2))
    denominator = np.sqrt(np.sum(old_missing**2))
    return bool(delta / (denominator + _EPS) < threshold)


def softimpute(
    X: npt.NDArray,
    shrinkage_value: Optional[float] = None,
    convergence_threshold: float = 0.001,
    max_iters: int = 100,
    max_rank: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    normalize: bool = True,
) -> npt.NDArray:
    """SoftImpute imputation method.

    Repeatedly replaces the missing entries with the current low-rank estimate
    and re-estimates by soft-thresholding the singular values, which solves a
    nuclear-norm regularized completion problem (Mazumder et al., 2010).

    Args:
        X (npt.NDArray): N x T input data matrix with missing values as
            ``np.nan``.
        shrinkage_value (Optional[float]): Amount subtracted from each singular
            value. Defaults to ``max(singular values) / 50``.
        convergence_threshold (float): Stop once the relative change in the
            imputed entries falls below this.
        max_iters (int): Maximum number of iterations.
        max_rank (Optional[int]): Cap on the rank of the estimate. Defaults to
            no cap.
        min_value (Optional[float]): Clip imputed values below this.
        max_value (Optional[float]): Clip imputed values above this.
        normalize (bool): Whether to bi-scale rows and columns before imputing
            and undo it afterwards. Matches passing a ``BiScaler`` normalizer.

    Raises:
        ValueError: If ``X`` is not two-dimensional, has no observed entries, or
            contains a row or column with no observed entries at all.

    Returns:
        npt.NDArray: The imputed data matrix.

    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-dimensional matrix, got shape {X.shape}")

    missing = np.isnan(X)
    observed = ~missing
    if not observed.any():
        raise ValueError("Cannot impute a matrix with no observed entries")

    # A row or column with nothing observed carries no information for this
    # estimator, and would come back as an arbitrary value near the global
    # mean. Refuse rather than return something meaningless. Callers should
    # drop such rows and columns, or use a method that can borrow across them.
    empty_rows = int(np.count_nonzero(~observed.any(axis=1)))
    empty_columns = int(np.count_nonzero(~observed.any(axis=0)))
    if empty_rows or empty_columns:
        raise ValueError(
            f"{empty_rows} rows and {empty_columns} columns have no observed "
            "values, so they cannot be imputed. Drop them before calling "
            "softimpute."
        )

    if not missing.any():
        return X.copy()

    n_rows, n_cols = X.shape
    if normalize:
        working, row_center, col_center, row_scale, col_scale = _biscale(X, observed)
    else:
        # Identity transform, so the un-normalizing step below is a no-op.
        working = X
        row_center, col_center = np.zeros(n_rows), np.zeros(n_cols)
        row_scale, col_scale = np.ones(n_rows), np.ones(n_cols)

    filled = np.where(missing, 0.0, np.nan_to_num(working))

    if shrinkage_value is None:
        largest_singular_value = np.linalg.svd(filled, compute_uv=False)[0]
        shrinkage_value = float(largest_singular_value) / _DEFAULT_SHRINKAGE_DIVISOR

    for _ in range(max_iters):
        reconstructed = _soft_threshold_svd(filled, shrinkage_value, max_rank)
        if min_value is not None or max_value is not None:
            reconstructed = np.clip(reconstructed, min_value, max_value)

        has_converged = _converged(
            filled, reconstructed, missing, convergence_threshold
        )
        filled[missing] = reconstructed[missing]
        if has_converged:
            break

    if normalize:
        filled = filled * np.outer(row_scale, col_scale) + (
            row_center[:, None] + col_center[None, :]
        )
        # Observed entries are returned exactly as supplied; only the missing
        # ones are estimates.
        filled[observed] = X[observed]

    return filled
