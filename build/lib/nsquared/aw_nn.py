"""A function to return the Adaptively-weighted Nearest Neighbors imputer."""

from typing import Optional

# from nsquared.weighted_estimation_methods import AWNNEstimator
from nsquared.data_types import Scalar
from nsquared.nnimputer import NearestNeighborImputer
from nsquared.estimation_methods import AWNNEstimator


def aw_nn(
    distance_threshold: Optional[float] = 0,
    delta: float = 1,
    convergence_threshold: float = 1e-4,
    max_iterations: int = 10,
    noise_variance: Optional[float] = None,
) -> NearestNeighborImputer:
    """Create a AWNN imputer."""
    # distance_threshold defaults to 0 and is wholly ignored by the AWNN approach.
    estimator = AWNNEstimator(
        delta=delta,
        convergence_threshold=convergence_threshold,
        max_iterations=max_iterations,
        noise_variance=noise_variance,
    )
    data_type = Scalar()
    return NearestNeighborImputer(estimator, data_type, distance_threshold)
