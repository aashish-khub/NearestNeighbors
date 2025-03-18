"""MCAR simulation data

TODO: remove commented out code
"""

import numpy as np
from typing import Tuple


def gendata_lin_mcar(
    N: int, T: int, p: float, seed: int, r: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates data using a bilinear model with uniform latent factors of dimension r.

    Parameters
    ----------
    N: int
        Number of users.
    T: int
        Number of time periods.
    p: float
        Probability of an entry being observed.
    seed: int
        Random seed for reproducibility.
    r: int, optional
        Dimension of latent factors. Default is 4.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Data (np.ndarray): Generated data matrix of shape (N, T).
        Theta (np.ndarray): Underlying true data matrix of shape (N, T).
        Masking (np.ndarray): Masking matrix indicating observed entries of shape (N, T).

    """
    # TODO: pass in random number generator rather than seed
    np.random.seed(seed=seed)
    ## Data Matrix (N * T)
    Data: np.ndarray = np.zeros((N, T))

    # user_range = 0.9
    # time_range = 0.9

    # user_std = 0.3
    # time_std = 0.3

    U = np.random.uniform(-1, 1, size=(N, r))  # * user_range * 2 - user_range
    V = np.random.uniform(-1, 1, size=(T, r))  # * time_range * 2 - time_range
    Y = np.matmul(U, V.transpose())
    # 1/np.sqrt(r) * np.matmul(U, V.transpose())

    # a = np.random.normal(0, user_std, size=N)
    # b = np.random.normal(0, time_std, size=T)
    # eps = np.random.normal(0, 0.05, size=(N, T))

    # treatment effect
    # delta(i,j) = a(i) + b(t) + eps(i,t)

    # aa = np.broadcast_to(a.reshape(N,1), (N,T))
    # bb = np.broadcast_to(b, (N,T))
    # delta = aa + bb + eps
    # Y1 = Y0 + delta

    # Y1 += np.random.normal(0, 0.001, size=(N,T))
    # gaussian noise
    Theta: np.ndarray = Y
    Y += np.random.normal(0, 0.001, size=(N, T))
    # TODO: clean up this code
    Masking: np.ndarray = np.zeros((N, T))

    Masking = np.reshape(np.random.binomial(1, p, (N * T)), (N, T))

    # Data[Masking == 1] = Y1[Masking == 1]
    # Data[Masking == 0] = Y0[Masking == 0]
    Data = np.array(Y)
    return Data, Theta, Masking


def gendata_nonlin_mcar(
    N: int, T: int, p: float, seed: int, non_lin: str = "expit", r: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates data using a nonlinear model (default: expit) with uniform latent factors of dimension r.

    Parameters
    ----------
    N: int
        Number of users.
    T: int
        Number of time periods.
    p: float
        Probability of an entry being observed.
    seed: int
        Random seed for reproducibility.
    non_lin: str, optional
        Nonlinear function to apply. Options are "expit", "tanh", "sin", "cubic", "sinh". Default is "expit".
    r: int, optional
        Dimension of latent factors. Default is 4.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Data (np.ndarray): Generated data matrix of shape (N, T).
        Theta (np.ndarray): Latent variable matrix of shape (N, T).
        Masking (np.ndarray): Masking matrix indicating observed entries of shape (N, T).

    """
    # TODO: pass in random number generator rather than seed
    np.random.seed(seed=seed)

    def expit(x: np.ndarray) -> np.ndarray:
        return np.exp(x) / (1 + np.exp(x))

    Data = np.zeros((N, T))
    U = np.random.uniform(-1, 1, size=(N, r))
    V = np.random.uniform(-1, 1, size=(T, r))
    if non_lin == "expit":
        Y = expit(np.matmul(U, V.transpose()))
    elif non_lin == "tanh":
        Y = np.tanh(np.matmul(U, V.transpose()))
    elif non_lin == "sin":
        Y = np.sin(np.matmul(U, V.transpose()))
    elif non_lin == "cubic":
        Y = (np.matmul(U, V.transpose())) ** 3
    elif non_lin == "sinh":
        Y = np.sinh(np.matmul(U, V.transpose()))
    else:
        raise ValueError(
            "non_lin must be one of 'expit', 'tanh', 'sin', 'cubic', or 'sinh'."
        )
    Theta: np.ndarray = Y
    Y += np.random.normal(0, 0.001, size=(N, T))

    # TODO: clean up this code
    Masking: np.ndarray = np.zeros((N, T))

    Masking: np.ndarray = np.reshape(np.random.binomial(1, p, (N * T)), (N, T))

    # Data[Masking == 1] = Y1[Masking == 1]
    # Data[Masking == 0] = Y0[Masking == 0]
    Data: np.ndarray = np.array(Y)
    return Data, Theta, Masking


def gendata_dist_mcar(N: int, T: int, n: int, d: int, p: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates multivariate Gaussian data of multiple measurements with latent dimension r = 2.

    Args:
        N (int): Number of users.
        T (int): Number of time periods.
        n (int): Number of samples per distribution.
        d (int): Dimension of the data.
        p (float): Probability of an entry being observed.
        seed (int): Random seed for reproducibility.

    Returns:
        Data (np.ndarray): Generated data matrix of shape (N, T, n, d).
        Masking (np.ndarray): Masking matrix indicating observed entries of shape (N, T).
        True Mean (np.ndarray): True mean of the data of shape (N, T, d).
        True Covariance (np.ndarray): True covariance of the data of shape (N, T, d, d).

    """
    np.random.seed(seed=seed)

    ## Data Matrix (N * T * n * d)
    Data = np.zeros((N, T, n, d))
    true_Mean = np.zeros((N, T, d))
    true_Cov = np.zeros((N, T, d, d))

    u_1 = np.random.uniform(-1, 1, N)
    u_2 = np.random.uniform(0.2, 1, N)

    v_1 = np.random.uniform(-2, 2, T)
    v_2 = np.random.uniform(0.5, 2, T)

    even_ones = np.repeat([0, 1], int(d/2))
    odd_ones = np.repeat([1, 0], int(d/2))

    for i in range(N):
        for t in range(T):
            m_it = u_1[i]*v_1[t]*(even_ones - odd_ones)
            c_it = np.diag(u_2[i]*v_2[t]*(0.5*even_ones + odd_ones))
            true_Mean[i, t, :] = m_it
            true_Cov[i, t, :, :] = c_it
            dat_mat = np.random.multivariate_normal(m_it, c_it, size=n)
            Data[i, t, :, :] = dat_mat

    Masking = np.zeros((N, T))
    Masking = np.reshape(np.random.binomial(1, p, (N*T)), (N, T))

    return Data, Masking, true_Mean, true_Cov