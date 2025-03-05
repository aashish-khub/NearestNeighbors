"""Kernel function implementations

TODO: add linear, square, and exponential kernels from https://github.com/calebchin/DistributionalNearestNeighbors/blob/main/src/kernel_nn.py
"""

import numpy as np

def laplace(dists: np.ndarray, eta: float) -> np.ndarray:
    """Compute the Laplace kernel given a distance matrix.

    Args:
        dists (np.ndarray): Distance matrix.
        eta (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    assert eta > 0
    dists = np.where(dists < 0, 0, dists)
    gamma = 1.0 / eta
    dists *= -gamma
    kernel_mat = np.exp(dists)
    return kernel_mat


def gaussian(dists: np.ndarray, eta: float) -> np.ndarray:
    """Compute the Gaussian kernel given a distance matrix.

    Args:
        dists (np.ndarray): Distance matrix.
        eta (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    assert eta > 0
    dists = np.where(dists < 0, 0, dists)
    gamma = 0.5 / eta**2
    dists *= -gamma
    kernel_mat = np.exp(dists)
    return kernel_mat


def singular(dists: np.ndarray, eta: float) -> np.ndarray:
    """Compute the Singular kernel given a distance matrix.

    Args:
        dists (np.ndarray): Distance matrix.
        eta (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    dists /= eta
    a = 0.49
    # replace divide by zero with nan
    dists = np.where(dists == 0, np.nan, dists)
    kernel_matrix = np.power(dists, -a) * np.maximum(0, 1 - dists) ** 2
    return kernel_matrix


def singular_box(
    samples: np.ndarray, centers: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Compute the Singular Box kernel between samples and centers.
def singular_box(dists: np.ndarray, eta: float) -> np.ndarray:
    """Compute the Singular Box kernel given a distance matrix.

    Args:
        dists (np.ndarray): Distance matrix.
        eta (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    assert eta > 0
    dists /= eta
    a = 0.49
    dists = np.where(dists == 0, np.nan, dists)
    kernel_mat = np.power(dists, -a) * np.where(dists <= 1, 1, 0)
    return kernel_mat


def box(dists: np.ndarray, eta: float) -> np.ndarray:
    """Compute the Box kernel given a distance matrix.

    Args:
        dists (np.ndarray): Distance matrix.
        eta (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix with values 1 where the distance is within the bandwidth, otherwise 0.

    """
    assert eta > 0
    dists /= eta
    kernel_mat = np.where(dists <= 1, 1, 0)
    return kernel_mat


def epanechnikov(
    samples: np.ndarray, centers: np.ndarray, bandwidth: float
) -> np.ndarray:
    r"""Compute the Epanechnikov kernel between samples and centers.
def epanechnikov(dists: np.ndarray, eta: float) -> np.ndarray:
    r"""Compute the Epanechnikov kernel given a distance matrix.
        Epanechnikov kernel: \kappa(u) = 3/4 * (1 - u^2) for u in [-1, 1],
        where u = dists / eta

    Args:
        dists (np.ndarray): Distance matrix.
        eta (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix with values computed using the Epanechnikov kernel function.

    """
    assert eta > 0
    dists /= eta
    kernel_mat = np.where(dists <= 1, 0.75 * (1 - dists**2), 0)
    return kernel_mat


def wendland(dists: np.ndarray, eta: float) -> np.ndarray:
    r"""Compute the Wendland kernel given a distance matrix.
        Wendland kernel: \kappa(u) = (1 - u)_+ for u in [0, 1],
        where u = dists / eta

    Args:
        dists (np.ndarray): Distance matrix.
        eta (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix with values computed using the Wendland kernel function.
    
    """
    assert eta > 0
    dists /= eta
    kernel_mat = np.where(dists <= 1, 1 - dists, 0)
    return kernel_mat
