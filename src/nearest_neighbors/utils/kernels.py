"""Kernel function implementations

TODO: add linear, square, and exponential kernels from https://github.com/calebchin/DistributionalNearestNeighbors/blob/main/src/kernel_nn.py
"""

import numpy as np


def euclidean_distances(
    samples: np.ndarray, centers: np.ndarray, squared: bool = True
) -> np.ndarray:
    """Compute the Euclidean distances between samples and centers.

    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        squared (bool): If True, return squared distances. Default is True.

    Returns:
        np.ndarray: Matrix of distances between samples and centers.

    """
    samples_norm = np.sum(samples**2, axis=1, keepdims=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = np.sum(centers**2, axis=1, keepdims=True)
    centers_norm = np.reshape(centers_norm, (1, -1))
    distances = samples @ centers.T
    distances *= -2
    distances = distances + samples_norm + centers_norm
    if not squared:
        distances = np.where(distances < 0, 0, distances)
        distances = np.sqrt(distances)
    return distances


def laplace(samples: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute the Laplace kernel between samples and centers.

    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat = np.where(kernel_mat < 0, 0, kernel_mat)
    gamma = 1.0 / bandwidth
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat


def gaussian(samples: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute the Gaussian kernel between samples and centers.

    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=True)
    kernel_mat = np.where(kernel_mat < 0, 0, kernel_mat)
    gamma = 0.5 / bandwidth**2
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat


def sobolev(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Compute the Sobolev kernel between samples and centers.

    Args:
        samples (np.ndarray): An array of sample points.
        centers (np.ndarray): An array of center points.

    Returns:
        np.ndarray: The computed Sobolev kernel matrix.

    """
    return 1 + np.minimum(samples, centers.T)


def singular(samples: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute the Singular kernel between samples and centers.

    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    a = 0.49
    # replace divide by zero with nan
    kernel_mat = np.where(kernel_mat == 0, np.nan, kernel_mat)
    return np.power(kernel_mat, -a) * np.maximum(0, 1 - kernel_mat) ** 2


def singular_box(
    samples: np.ndarray, centers: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Compute the Singular Box kernel between samples and centers.

    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    a = 0.49
    kernel_mat = np.where(kernel_mat == 0, np.nan, kernel_mat)
    kernel_mat = np.power(kernel_mat, -a) * np.where(kernel_mat <= 1, 1, 0)
    return kernel_mat


def box(samples: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    r"""Box kernel: \kappa(u) = 1 for u in [-1, 1],
    where u = ||x - y||_2 / bandwidth
    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix with values 1 where the distance is within the bandwidth, otherwise 0.

    """
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    # indicator for kernel_mat <= 1
    kernel_mat = np.where(kernel_mat <= 1, 1, 0)
    return kernel_mat


def epanechnikov(
    samples: np.ndarray, centers: np.ndarray, bandwidth: float
) -> np.ndarray:
    r"""Compute the Epanechnikov kernel between samples and centers.
        Epanechnikov kernel: \kappa(u) = 3/4 * (1 - u^2) for u in [-1, 1],
        where u = ||x - y||_2 / bandwidth

    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix with values computed using the Epanechnikov kernel function.

    """
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    # indicator for kernel_mat <= 1
    kernel_mat = np.where(kernel_mat <= 1, 0.75 * (1 - kernel_mat**2), 0)
    return kernel_mat


def wendland(samples: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    r"""Compute the Wedland kernel between samples and centers.
        Wedland kernel: \kappa(u) = (1 - u)_+ for u in [-1, 1],
    where u = ||x - y||_2 / bandwidth

    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix with values computed using the Epanechnikov kernel function.

    """
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    # indicator for kernel_mat <= 1
    kernel_mat = np.where(kernel_mat <= 1, 1 - kernel_mat, 0)
    return kernel_mat


# With feature matrix M
def euclidean_distances_M(
    samples: np.ndarray, centers: np.ndarray, M: np.ndarray, squared: bool = True
) -> np.ndarray:
    """Returns the Euclidean distances between samples and centers with feature matrix M.

    Args:
        samples (np.ndarray): shape(n1, d)
        centers (np.ndarray): shape(n2, d)
        M (np.ndarray): shape(d, d)
        squared (bool): If True, return squared distances. Default is True.

    Returns:
        np.ndarray: Matrix of distances between samples and centers with feature matrix M.

    """
    samples_norm2 = ((samples @ M) * samples).sum(-1)

    if samples is centers:
        centers_norm2 = samples_norm2
    else:
        centers_norm2 = ((centers @ M) * centers).sum(-1)

    distances = -2 * (samples @ M) @ centers.T
    distances += samples_norm2.reshape(-1, 1)
    distances += centers_norm2

    if not squared:
        # assert distances are positive
        tol = -0.01
        assert len(distances[distances < tol]) == 0, f"{distances[distances < tol]}"
        distances = np.sqrt(np.clip(distances, a_min=0, a_max=None))

    return distances


def laplace_M(
    samples: np.ndarray, centers: np.ndarray, M: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Compute the Laplace kernel between samples and centers with feature matrix M.

    Args:
        samples (np.ndarray): shape(n1, d)
        centers (np.ndarray): shape(n2, d)
        M (np.ndarray): shape(d, d)
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    gamma = 1.0 / bandwidth
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat


def gaussian_M(
    samples: np.ndarray, centers: np.ndarray, M: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Compute the Gaussian kernel between samples and centers with feature matrix M.

    Args:
        samples (np.ndarray): Array of sample points.
        centers (np.ndarray): Array of center points.
        M (np.ndarray): Feature matrix.
        bandwidth (float): Bandwidth parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix.

    """
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=True)
    gamma = 1.0 / (2 * bandwidth**2)
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat
