"""Confounded Staggered adoption simulation

T: Number of cols
n: Number of measurements per entry
d: Dimension of measurements
beta: Degree of confounding

"""

import numpy as np
import math as math
from typing import Tuple


def expit(x: float) -> float:
    """Compute the logistic sigmoid function of x.

    Args:
        x: Input value

    Returns:
        float: 1 / (1 + exp(-x))

    """
    return 1 / (1 + np.exp(-x))


def gendata_s_adopt(
    N: int, T: int, n: int, d: int, beta: Tuple[float, float], seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates confounded/staggered adoption "Gaussian" data for distributional NN, with latent dimension r = 2

    Args:
        N: Number of rows (even)
        T: Number of columns
        n: Number of measurements per entry
        d: Dimension of measurements (even)
        beta: vector of two fractions between (0, 1)
        seed: random seed

    Returns:
        Data: (N, T, n, d) array of data
        Masking: (N, T) array of missing indicators
        true_Mean: (N, T, d) array of true means
        true_Cov: (N, T, d, d) array of true covariances

    """
    np.random.seed(seed=seed)

    Data = np.zeros((N, T))
    true_Mean = np.zeros((N, T))
    true_Cov = np.zeros((N, T))

    u_1 = np.random.uniform(-1, 1, N)
    u_2 = np.random.uniform(0.2, 1, N)

    v_1 = np.random.uniform(-2, 2, T)
    v_2 = np.random.uniform(0.5, 2, T)

    even_ones = np.repeat([0, 1], int(d / 2))
    odd_ones = np.repeat([1, 0], int(d / 2))

    for i in range(N):
        for t in range(T):
            m_it = u_1[i] * v_1[t] * (even_ones - odd_ones)
            c_it = np.diag(u_2[i] * v_2[t] * (0.5 * even_ones + odd_ones))
            true_Mean[i, t] = m_it
            true_Cov[i, t] = c_it
            dat_mat = np.random.multivariate_normal(m_it, c_it, size=n)
            Data[i, t] = dat_mat

    Masking = np.zeros((N, T))
    pre_Masking = np.zeros((N, T))

    g1_inds = np.arange(0, N // 3)
    g2_inds = np.arange(N // 3, 2 * N // 3)
    g3_inds = np.arange(2 * N // 3, N)

    gamma_1 = [2, 0.7, 1, 0.7]
    gamma_2 = [2, 0.2, 1, 0.2]

    T1_lower = math.floor(T ** beta[0])
    T2_lower = math.floor(T ** beta[1])

    for i in range(N):
        if i in g1_inds:
            pre_Masking[i, :] = np.concatenate(
                (np.ones(T1_lower), np.zeros(T - T1_lower))
            )
            for t in range(T - T1_lower):
                pre_Masking[i, (t + T1_lower)] = np.random.binomial(
                    1,
                    expit(
                        gamma_1[0]
                        + (0.99**t) * gamma_1[1] * u_1[i - 1]
                        + gamma_1[2] * u_1[i]
                        + (0.99**t) * gamma_1[3] * u_1[i + 1]
                    ),
                    1,
                )
            pre_A = pre_Masking[i, :]
            if len([i for i in range(len(pre_A)) if pre_A[i] == 0]) == 0:
                Masking[i, :] = pre_A
            elif len([i for i in range(len(pre_A)) if pre_A[i] == 0]) > 0:
                adopt_time = min([i for i in range(len(pre_A)) if pre_A[i] == 0])
                Masking[i, :] = np.concatenate(
                    (np.ones(adopt_time), np.zeros(T - adopt_time))
                )
        elif i in g2_inds:
            pre_Masking[i, :] = np.concatenate(
                (np.ones(T2_lower), np.zeros(T - T2_lower))
            )
            for t in range(T - T2_lower):
                pre_Masking[i, (t + T2_lower)] = np.random.binomial(
                    1,
                    expit(
                        gamma_2[0]
                        + (1.01**t) * gamma_2[1] * u_1[i - 1]
                        + gamma_2[2] * u_1[i]
                        + (1.01**t) * gamma_2[3] * u_1[i + 1]
                    ),
                    1,
                )
            pre_A = pre_Masking[i, :]
            if len([i for i in range(len(pre_A)) if pre_A[i] == 0]) == 0:
                Masking[i, :] = pre_A
            elif len([i for i in range(len(pre_A)) if pre_A[i] == 0]) > 0:
                adopt_time = min([i for i in range(len(pre_A)) if pre_A[i] == 0])
                Masking[i, :] = np.concatenate(
                    (np.ones(adopt_time), np.zeros(T - adopt_time))
                )
        elif i in g3_inds:
            Masking[i, :] = np.ones(T)

    return Data, Masking, true_Mean, true_Cov
