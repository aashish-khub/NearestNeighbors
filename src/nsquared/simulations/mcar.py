"""MCAR (missing completely at random) simulation."""

import numpy as np
import numpy.typing as npt
from typing import Tuple

from .latent_factors import apply_nonlinear_transform

# Observation noise used by the self-contained generators below.
NOISE_STDDEV = 0.001


def make_mcar_mask(num_rows: int, num_cols: int, miss_prob: float) -> npt.NDArray:
    """Draw a missingness mask with every entry missing independently.

    Args:
        num_rows (int): Number of rows N.
        num_cols (int): Number of columns T.
        miss_prob (float): Probability that any given entry is missing.

    Returns:
        npt.NDArray: Boolean array of shape ``(N, T)``, True where the entry is
            missing.

    """
    return np.random.binomial(1, miss_prob, size=(num_rows, num_cols)) == 1


def gendata_lin_mcar(
    N: int,
    T: int,
    p: float,
    seed: int | None = None,
    r: int = 4,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data from a bilinear model with uniform latent factors.

    Args:
        N (int): Number of users.
        T (int): Number of time periods.
        p (float): Probability of an entry being observed.
        seed (int | None): Random seed. Ignored when ``rng`` is given.
        r (int): Dimension of the latent factors.
        rng (np.random.Generator | None): Generator to draw from. Supplying one
            avoids touching NumPy's global random state.

    Returns:
        Tuple: ``Data`` of shape (N, T), the noiseless ``Theta`` of the same
            shape, and a ``Masking`` array that is 1 where an entry is observed.

    """
    rng = rng if rng is not None else np.random.default_rng(seed)

    row_factors = rng.uniform(-1, 1, size=(N, r))
    col_factors = rng.uniform(-1, 1, size=(T, r))
    theta = row_factors @ col_factors.T

    data = theta + rng.normal(0, NOISE_STDDEV, size=(N, T))
    masking = rng.binomial(1, p, size=(N, T))
    return data, theta, masking


def gendata_nonlin_mcar(
    N: int,
    T: int,
    p: float,
    seed: int | None = None,
    non_lin: str = "expit",
    r: int = 4,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data from a nonlinear model with uniform latent factors.

    Args:
        N (int): Number of users.
        T (int): Number of time periods.
        p (float): Probability of an entry being observed.
        seed (int | None): Random seed. Ignored when ``rng`` is given.
        non_lin (str): Nonlinearity applied to the bilinear signal. One of
            ``expit``, ``tanh``, ``sin``, ``cubic``, ``sinh``.
        r (int): Dimension of the latent factors.
        rng (np.random.Generator | None): Generator to draw from. Supplying one
            avoids touching NumPy's global random state.

    Returns:
        Tuple: ``Data`` of shape (N, T), the noiseless ``Theta`` of the same
            shape, and a ``Masking`` array that is 1 where an entry is observed.

    """
    rng = rng if rng is not None else np.random.default_rng(seed)

    row_factors = rng.uniform(-1, 1, size=(N, r))
    col_factors = rng.uniform(-1, 1, size=(T, r))
    theta = apply_nonlinear_transform(row_factors @ col_factors.T, non_lin)

    data = theta + rng.normal(0, NOISE_STDDEV, size=(N, T))
    masking = rng.binomial(1, p, size=(N, T))
    return data, theta, masking
