"""A function to return the Star NN imputer."""

from typing import Optional

# from nearest_neighbors.weighted_estimation_methods import StarNNEstimator
from nearest_neighbors.data_types import Scalar
from nearest_neighbors.nnimputer import NearestNeighborImputer
from nearest_neighbors.estimation_methods import StarNNEstimator


def star_nn(
    distance_threshold: Optional[float] = 0,
    delta: float = 1,
    convergence_threshold: float = 1e-4,
    max_iterations: int = 10,
    noise_variance: Optional[float] = None,
) -> NearestNeighborImputer:
    """Create a Star Nearest Neighbor imputer."""
    # distance_threshold defaults to 0 and is wholly ignored by the StarNN approach.
    estimator = StarNNEstimator(
        delta=delta,
        convergence_threshold=convergence_threshold,
        max_iterations=max_iterations,
        noise_variance=noise_variance,
    )
    data_type = Scalar()
    return NearestNeighborImputer(estimator, data_type, distance_threshold)
