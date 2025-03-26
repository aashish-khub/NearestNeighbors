import numpy as np


def usvt(A: np.ndarray, eta: float = 0.0001) -> np.ndarray:
    """Compute the USVT matrix imputations

    Args:
    ----
    A : a N x T matrix with missing values denoted as nan
    eta : a small positive (or 0) that affects the singular value threshold.
            by default 0

    Returns:
    -------
    A_hat : the USVT estimate of A

    """
    # later should generalize for all matrices (i.e. transpose if m > n)

    N, T = A.shape

    # data prep
    # of observed values,
    a = np.nanmin(A)
    b = np.nanmax(A)

    A_scaled = (A - ((a + b) / 2)) / ((b - a) / 2)
    Y_scaled = np.where(np.isnan(A_scaled), 0, A_scaled)

    p_hat = 1 - (np.count_nonzero(np.isnan(A)) / np.size(A))

    # currently follow Chatterjee 2015 threshold value
    threshold = (2 + eta) * np.sqrt(T * p_hat)

    u, s, vt = np.linalg.svd(Y_scaled, full_matrices=False)

    s_mask = s >= threshold
    W = (u[:, s_mask] @ np.diag(s[s_mask]) @ vt[s_mask, :]) / p_hat

    # cap to [-1, 1]
    W = W.clip(-1, 1)
    # rescale:
    res = (W * ((b - a) / 2)) + (a + b) / 2

    return res
