"""Latent-factor data generation for synthetic matrix completion experiments.

These are the building blocks behind
``nsquared.datasets.synthetic_data.SyntheticDataLoader``, factored out so that
the generative model can be used, tested, and extended on its own rather than
being locked inside a data loader.

The model is the standard one for the matrix completion literature: draw row
factors :math:`u_i` and column factors :math:`v_t`, combine them into a signal
matrix, optionally distort it, then add noise.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt

# Nonlinear distortions that can be applied to the signal matrix.
NONLINEAR_TRANSFORMS = ("", "expit", "tanh", "sin", "cubic", "sinh")
# Ways row and column factors can be combined into a signal.
COMBINATION_MODELS = ("multiplicative", "additive")


def generate_latent_factors(
    num_rows: int, num_cols: int, dimensionality: int = 4
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Draw row and column latent factors uniformly from [-0.5, 0.5].

    Args:
        num_rows (int): Number of rows (units) N.
        num_cols (int): Number of columns (time periods) T.
        dimensionality (int): Latent dimension r.

    Returns:
        Tuple: Row factors of shape ``(num_rows, r)`` and column factors of
            shape ``(num_cols, r)``.

    """
    row_factors = np.random.uniform(size=(num_rows, dimensionality)) - 0.5
    col_factors = np.random.uniform(size=(num_cols, dimensionality)) - 0.5
    return row_factors, col_factors


def combine_latent_factors(
    row_factors: npt.NDArray,
    col_factors: npt.NDArray,
    model: str = "multiplicative",
    rho: float = 0.5,
) -> npt.NDArray:
    """Combine row and column factors into a signal matrix.

    Two models are supported:

    - ``"multiplicative"``: the usual bilinear signal, ``Y = U V^T``.
    - ``"additive"``: the Holder-continuous signal
      ``Y[i, t] = sum_k |u[i, k] + v[t, k]|^rho * sign(u[i, k] + v[t, k])``,
      which admits a nonlinear relationship between the factors. See
      https://doi.org/10.48550/arXiv.2411.12965.

    Args:
        row_factors (npt.NDArray): Row factors, shape ``(N, r)``.
        col_factors (npt.NDArray): Column factors, shape ``(T, r)``.
        model (str): One of ``COMBINATION_MODELS``.
        rho (float): Holder exponent, used by the additive model only.

    Raises:
        ValueError: If ``model`` is not a supported combination model.

    Returns:
        npt.NDArray: Signal matrix of shape ``(N, T)``.

    """
    if model == "multiplicative":
        return row_factors @ col_factors.T
    if model == "additive":
        summed = row_factors[:, np.newaxis] + col_factors
        return (np.abs(summed) ** rho * np.sign(summed)).sum(axis=2)
    raise ValueError(f"model must be one of {COMBINATION_MODELS}, got {model!r}")


def apply_nonlinear_transform(matrix: npt.NDArray, kind: str = "") -> npt.NDArray:
    """Apply a nonlinear distortion to a signal matrix.

    Args:
        matrix (npt.NDArray): Signal matrix.
        kind (str): One of ``NONLINEAR_TRANSFORMS``. The empty string, the
            default, leaves the matrix untouched.

    Raises:
        ValueError: If ``kind`` is not a supported transform.

    Returns:
        npt.NDArray: The transformed matrix. The input is not modified.

    """
    if kind == "":
        return matrix
    if kind == "expit":
        # Evaluated branchwise so that neither tail overflows: exp(-x) blows up
        # for very negative x, and exp(x) for very positive x.
        result = np.empty(matrix.shape, dtype=float)
        positive = matrix >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-matrix[positive]))
        exponentiated = np.exp(matrix[~positive])
        result[~positive] = exponentiated / (1.0 + exponentiated)
        return result
    if kind == "tanh":
        return np.tanh(matrix)
    if kind == "sin":
        return np.sin(matrix)
    if kind == "cubic":
        return matrix**3
    if kind == "sinh":
        return np.sinh(matrix)
    raise ValueError(f"kind must be one of {NONLINEAR_TRANSFORMS}, got {kind!r}")


def noise_scale_for_snr(signal: npt.NDArray, snr: float) -> float:
    """Return the noise standard deviation giving a target signal-to-noise ratio.

    Args:
        signal (npt.NDArray): The noiseless signal matrix.
        snr (float): Desired ratio of signal variance to noise variance.

    Returns:
        float: The standard deviation to use for the additive noise.

    """
    return float(np.sqrt(np.mean(signal**2) / snr))


def add_gaussian_noise(signal: npt.NDArray, stddev: float) -> npt.NDArray:
    """Add i.i.d. centered Gaussian noise to a signal matrix.

    Args:
        signal (npt.NDArray): The noiseless signal matrix.
        stddev (float): Standard deviation of the noise.

    Returns:
        npt.NDArray: A new noisy matrix. The input is not modified.

    """
    return signal + stddev * np.random.normal(size=signal.shape)
