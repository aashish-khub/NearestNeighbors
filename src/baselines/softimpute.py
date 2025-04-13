"""SoftImpute algorithm for matrix completion.
This algorithm is based on the paper:

Hastie, T., Mazumder, R., Lee, J. D., & Zadeh, R. (2015).
Matrix completion and low-rank SVD via fast alternating least squares.
The Journal of Machine Learning Research, 16(1), 3367-3402.

The implementation is based on the python package `fancyimpute` implementation.
https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/soft_impute.py
"""

import numpy as np
from typing import Optional
from sklearn.utils import check_array
from sklearn.utils.extmath import randomized_svd


def _residual(data: np.ndarray) -> np.ndarray:
    """Computes the residual of the data matrix.

    Args:
    ----
    data : np.ndarray
        The input data matrix.

    Returns:
    -------
    int
        The residual of the data matrix.

    """
    row_mean = np.nanmean(data, axis=1)
    col_mean = np.nanmean(data, axis=0)
    row_vars = np.nanvar(data, axis=1)
    row_vars[row_vars == 0] = 1.0
    col_vars = np.nanvar(data, axis=0)
    col_vars[col_vars == 0] = 1.0

    total = (
        np.sum(row_mean**2)
        + np.sum(col_mean**2)
        + np.sum(np.log(row_vars) ** 2)
        + np.sum(np.log(col_vars) ** 2)
    )
    return total


def _est_means(
    data: np.ndarray,
    obs: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    est_type: str = "row",
) -> np.ndarray:
    """Estimate the row means of the data matrix.

    Args:
    ----
    data : np.ndarray
        The input data matrix.
    obs : np.ndarray
        The mask matrix. An entry is observed in the data matrix if mask is 1 and unobserved if mask is 0.
    means : np.ndarray
        The column means of the data matrix.
    stds : np.ndarray
        The column standard deviations of the data matrix.
    est_type : str
        The type of estimation to perform. Options are "row" (row means) or "col" (col means).

    Returns:
    -------
    np.ndarray
        The estimated row means of the data matrix.

    """
    nr, nc = data.shape
    if est_type == "row":
        n_means = nc
        n_est = nr
        est_axis = 1
    else:
        n_means = nr
        n_est = nc
        est_axis = 0
        obs = obs.T

    data -= means.reshape((1, n_means))
    weights = 1.0 / stds
    data *= weights.reshape((1, n_means))
    est_means = np.zeros(n_est)
    est_resids = np.nansum(data, axis=est_axis)
    for i in range(n_est):
        mask = obs[i, :]
        sum_weights = np.sum(weights[mask])
        est_means[i] = est_resids[i] / sum_weights
    return est_means


def _est_stds(data: np.ndarray, stds: np.ndarray, est_type: str = "row") -> np.ndarray:
    """Estimate the row standard deviations of the data matrix.

    Args:
    ----
    data : np.ndarray
        The input data matrix.
    stds : np.ndarray
        The column standard deviations of the data matrix.
    est_type : str
        The type of estimation to perform. Options are "row" (row stds) or "col" (col stds).

    """
    nr, nc = data.shape
    if est_type == "row":
        n_stds = nc
        est_axis = 1
    else:
        n_stds = nr
        est_axis = 0

    vars = np.nanmean(data**2 / (stds**2).reshape((1, n_stds)), axis=est_axis)
    vars[vars == 0] = 1.0
    return np.sqrt(vars)


def _center(
    data: np.ndarray, row_means: np.ndarray, col_means: np.ndarray
) -> np.ndarray:
    """Center the data matrix by subtracting the row and column means.

    Args:
    ----
    data : np.ndarray
        The input data matrix.
    row_means : np.ndarray
        The row means of the data matrix.
    col_means : np.ndarray
        The column means of the data matrix.

    Returns:
    -------
    np.ndarray
        The centered data matrix.

    """
    data = data.copy()
    data -= row_means.reshape((data.shape[0], 1))
    data -= col_means.reshape((1, data.shape[1]))
    return data


def _rescale(
    data: np.ndarray, row_stds: np.ndarray, col_stds: np.ndarray
) -> np.ndarray:
    """Rescale the data matrix by dividing by the row and column standard deviations.

    Args:
    ----
    data : np.ndarray
        The input data matrix.
    row_stds : np.ndarray
        The row standard deviations of the data matrix.
    col_stds : np.ndarray
        The column standard deviations of the data matrix.

    """
    data = data.copy()
    data /= row_stds.reshape((data.shape[0], 1))
    data /= col_stds.reshape((1, data.shape[1]))
    return data


def _normalize(
    data: np.ndarray,
    mask: np.ndarray,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    max_iters: int = 100,
    tolerance: float = 0.001,
) -> np.ndarray:
    """Normalizes (center + scales) the data matrix by rows, cols, or rows and cols. The normalization is not done in place.

    Args:
    ----
    data : np.ndarray
        The input data matrix.
    mask : np.ndarray
        The mask matrix. An entry is observed in the data matrix if mask is 1 and unobserved if mask is 0.
    min_value : Optional[float]
        The minimum value to clip the data matrix to. By default, no clipping is done.
    max_value : Optional[float]
        The maximum value to clip the data matrix to. By default, no clipping is done.
    max_iters : int
        The maximum number of iterations to perform for the normalization.
    tolerance : float
        The tolerance for convergence. The algorithm stops when the relative change in the residual is less than this value.

    Returns:
    -------
    np.ndarray
        The normalized data matrix.

    """
    data = data.copy()
    data = np.clip(data, min_value, max_value)
    data[mask == 1] = np.nan
    data_rowmajor = np.asarray(data, order="C")
    data_colmajor = np.asarray(data, order="F")

    obs_rowmajor = ~np.isnan(data_rowmajor)
    obs_colmajor = ~np.isnan(data_colmajor)
    n_empty_rows = (np.sum(obs_rowmajor, axis=1) == 0).sum()
    n_empty_cols = (np.sum(obs_colmajor, axis=0) == 0).sum()
    if n_empty_rows > 0:
        raise ValueError(
            f"Cannot normalize rows with all missing values. Number of empty rows: {n_empty_rows}"
        )
    if n_empty_cols > 0:
        raise ValueError(
            f"Cannot normalize columns with all missing values. Number of empty columns: {n_empty_cols}"
        )
    # first, iteratively approxiate the row and col means and stds
    # init row means to 0 with unit variance, col means are obvs estimate
    row_means = np.zeros(data.shape[0])
    col_means = np.nanmean(data_rowmajor, axis=0)
    row_stds = np.ones(data.shape[0])
    col_stds = np.nanstd(data_rowmajor, axis=0)

    last_residual = _residual(data_rowmajor)
    for _ in range(max_iters):
        if last_residual == 0:
            break
        row_means = _est_means(data_rowmajor, obs_rowmajor, col_means, col_stds, "row")
        col_means = _est_means(data_colmajor, obs_colmajor, row_means, row_stds, "col")
        # center
        data_centered = _center(data, row_means, col_means)
        row_stds = _est_stds(data_centered, col_stds, "row")
        col_stds = _est_stds(data_centered, row_stds, "col")
        X_normed = _rescale(data_centered, row_stds, col_stds)
        new_residual = _residual(X_normed)
        delta_resid = last_residual - new_residual
        if delta_resid / last_residual < tolerance:
            break
        last_residual = new_residual

    data_norm = _center(data, row_means, col_means)
    data_norm = _rescale(data_norm, row_stds, col_stds)
    return data_norm


def _max_singular_value(data: np.ndarray) -> float:
    """Computes the maximum singular value of the data matrix.

    Args:
    ----
    data : np.ndarray
        The input data matrix.

    """
    _, s, _ = randomized_svd(data, 1, n_iter="auto", random_state=None)
    return s[0]


def _svd_step(
    data: np.ndarray,
    shrinkage_value: float,
    max_rank: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Performs a single step of the SVD algorithm.

    Args:
    ----
    data : np.ndarray
        The input data matrix.
    shrinkage_value : float
        The shrinkage value to apply to the singular values.
    max_rank : Optional[int]
        The maximum rank of the

    Returns:
    -------
    np.ndarray
        The reconstructed data matrix.
    int
        The rank of the reconstructed matrix.

    """
    if max_rank is not None:
        u, s, vt = randomized_svd(
            data, n_components=max_rank, n_iter="auto", random_state=None
        )
    else:
        u, s, vt = np.linalg.svd(data, full_matrices=False, compute_uv=True)
    s = np.maximum(s - shrinkage_value, 0)
    rank = (s > 0).sum()
    return u[:, :rank] @ np.diag(s[:rank]) @ vt[:rank, :], rank


def _converged(
    data: np.ndarray,
    mask: np.ndarray,
    data_recon: np.ndarray,
    convergence_threshold: float,
) -> bool:
    """Checks if the algorithm has converged.

    Args:
    ----
    data : np.ndarray
        The input data matrix.
    mask : np.ndarray
        The mask matrix. An entry is observed in the data matrix if mask is 1 and unobserved otherwise.
    data_recon : np.ndarray
        The reconstructed data matrix.
    convergence_threshold : float
        The convergence threshold.

    Returns:
    -------
    bool
        True if the algorithm has converged, False otherwise.

    """
    old_miss = data[mask == 0]
    new_miss = data_recon[mask == 0]
    ssd = np.sum((old_miss - new_miss) ** 2)
    norm = np.sqrt(np.sum(old_miss**2))
    if norm == 0:
        return False
    return (np.sqrt(ssd) / norm) < convergence_threshold


def softimpute(
    data: np.ndarray,
    mask: np.ndarray,
    shrinkage_value: Optional[float] = None,
    convergence_threshold: float = 0.001,
    max_iters: int = 100,
    max_rank: Optional[int] = None,
    init_fill_method: str = "zero",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    normalizer: Optional[str] = None,
) -> np.ndarray:
    """SoftImpute algorithm for matrix completion.

    Args:
    ----
    data : a N x T matrix with missing values denoted as nan
    mask : a N x T matrix with 1 if observed, 0 if missing
    shrinkage_value : Value to threshold the singular values in each iteration.
                      If None, it will be set to the maximum singular value of the initial matrix
                      divided by 50 (consistent with fancyimpute implemntation).
    convergence_threshold : Min ration difference between iterations before stopping
    max_iters : Maximum number of SVD iterations.
    max_rank : Maximum rank of the matrix. If not None, it will be used to perform a truncated SVD at each iteration
               with this value as the rank.
    init_fill_method : Method to fill the missing values in the initial matrix.
                       Currently supported options are "zero". Default is "zero".
    min_value : Minimum value to clip the matrix to.
    max_value : Maximum value to clip the matrix to.
    normalizer : Normalizer to use. Current options are "rowcol". Default is "rowcol".
                 If None, no normalization will be performed.

    """
    data = check_array(data, force_all_finite=False)  # type: ignore
    data = data.copy()
    if normalizer is not None:
        data = _normalize(data, mask, min_value, max_value)
    if init_fill_method == "zero":
        data = np.where(mask != 1, data, 0)
    else:
        raise ValueError(
            f"Unsupported init fill method: {init_fill_method}. Supported methods are: zero."
        )

    data_filled = data
    max_sv = _max_singular_value(data_filled)
    if shrinkage_value is None:
        # consistent with fancyimpute implementation
        shrinkage_value = max_sv / 50
    for i in range(max_iters):
        data_recon, rank = _svd_step(
            data_filled,
            shrinkage_value,
            max_rank=max_rank,
        )
        data_recon = np.clip(data_recon, min_value, max_value)
        converged = _converged(data_filled, mask, data_recon, convergence_threshold)
        if converged:
            break
        data_filled[mask == 0] = data_recon[mask == 0]

    return data_filled
