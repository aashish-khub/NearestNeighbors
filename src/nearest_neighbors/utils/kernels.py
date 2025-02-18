"""Kernel function implementations

TODO: add linear, square, and exponential kernels from https://github.com/calebchin/DistributionalNearestNeighbors/blob/main/src/kernel_nn.py
"""
import numpy as np

def euclidean_distances(samples, centers, squared=True):
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

def laplace(samples, centers, bandwidth):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat = np.where(kernel_mat < 0, 0, kernel_mat)
    gamma = 1. / bandwidth
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat

def gaussian(samples, centers, bandwidth):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=True)
    kernel_mat = np.where(kernel_mat < 0, 0, kernel_mat)
    gamma = 0.5 / bandwidth**2
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat

def sobolev(samples, centers):
    return 1 + np.minimum(samples, centers.T)

def singular(samples, centers, bandwidth):
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    a = 0.49
    # replace divide by zero with nan
    kernel_mat = np.where(kernel_mat == 0, np.nan, kernel_mat)
    return np.power(kernel_mat, -a) * np.maximum(0, 1 - kernel_mat)**2

def singular_box(samples, centers, bandwidth):
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    a = 0.49
    kernel_mat = np.where(kernel_mat == 0, np.nan, kernel_mat)
    kernel_mat = np.power(kernel_mat, -a) * np.where(kernel_mat <= 1, 1, 0)
    return kernel_mat

def box(samples, centers, bandwidth):
    """
    Box kernel: \kappa(u) = 1 for u in [-1, 1],
    where u = ||x - y||_2 / bandwidth
    """
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    # indicator for kernel_mat <= 1
    kernel_mat = np.where(kernel_mat <= 1, 1, 0)
    return kernel_mat

def epanechnikov(samples, centers, bandwidth):
    """
    Epanechnikov kernel: \kappa(u) = 3/4 * (1 - u^2) for u in [-1, 1],
    where u = ||x - y||_2 / bandwidth
    """
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    # indicator for kernel_mat <= 1
    kernel_mat = np.where(kernel_mat <= 1, 0.75 * (1 - kernel_mat**2), 0)
    return kernel_mat

def wendland(samples, centers, bandwidth):
    """
    Wendland kernel: \kappa(u) = (1 - u)_+ for u in [-1, 1],
    where u = ||x - y||_2 / bandwidth
    """
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat /= bandwidth
    # indicator for kernel_mat <= 1
    kernel_mat = np.where(kernel_mat <= 1, 1 - kernel_mat, 0)
    return kernel_mat

# With feature matrix M
def euclidean_distances_M(samples, centers, M, squared=True):
    """
    Args:
    - samples: shape(n1, d)
    - centers: shape(n2, d)
    - M: shape(d, d)
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
        tol =  -0.01
        assert len(distances[distances<tol])==0, f'{distances[distances<tol]}'
        distances = np.sqrt(np.clip(distances, a_min=0, a_max=None))

    return distances

def laplace_M(samples, centers, M, bandwidth):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    gamma = 1. / bandwidth
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat

def gaussian_M(samples, centers, M, bandwidth):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=True)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat