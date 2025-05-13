"""Implements the Kolmogorov-Smirnov distance.

References:
https://github.com/scipy/scipy/blob/e29dcb65a2040f04819b426a04b60d44a8f69c04/scipy/stats/_stats_py.py#L7920

"""

import numpy as np


def ks_distance(imputation: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov distance between the imputation and ground truth.

    Args:
        imputation (np.ndarray): The imputation
        ground_truth (np.ndarray): The ground truth

    Returns:
        float: The Kolmogorov-Smirnov distance

    """
    data1 = np.sort(imputation.flatten())
    data2 = np.sort(ground_truth.flatten())
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError("Data passed to ks_2samp must not be empty")

    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side="right") / n1
    cdf2 = np.searchsorted(data2, data_all, side="right") / n2
    cddiffs = cdf1 - cdf2
    return np.max(np.abs(cddiffs))
