"""A function to return the Star NN imputer."""

from typing import Optional
from nearest_neighbors.nnimputer import NearestNeighborImputer


def star_nn(
    noise_variance: Optional[float] = None,
    convergence_threshold: float = 1e-4,
    max_iterations: int = 10,
    delta: float = 0.05,
) -> NearestNeighborImputer:
    """Create a Star nearest neighbor imputer.
        No distance thresholds needed.

    Args:
        noise_variance (Optional[float]): The variance of the noise to be used in the imputation process. Defaults to None.
        convergence_threshold (float): The threshold for convergence during the iterative imputation process. Defaults to 1e-4.
        max_iterations (int): The maximum number of iterations allowed for the imputation process. Defaults to 10.

    Returns:
        NearestNeighborImputer: A two-sided nearest neighbor imputer.

    """
    from .estimation_methods import StarNNEstimator
    from .data_types import Scalar

    estimator = StarNNEstimator(
        delta=delta,
        noise_variance=noise_variance,
        convergence_threshold=convergence_threshold,
        max_iterations=max_iterations,
    )
    data_type = Scalar()
    imputer = NearestNeighborImputer(
        estimator, data_type, distance_threshold=0 # No distance thresholds needed for star_nn
    )
    return imputer
