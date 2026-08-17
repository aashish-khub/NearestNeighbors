"""A function to return the TSNN imputer.

More details on the Two Sided Nearest Neighbor (TSNN) method can be found in Section 3 of the following paper:
    Sadhukhan, T., Paul, M., Dwivedi, R. (2024).
    On adaptivity and minimax optimality of two-sided nearest neighbors. arXiv preprint arXiv:2411.12965.
"""

from .nnimputer import NearestNeighborImputer
from typing import Optional


def ts_nn(
    distance_threshold_row: Optional[float] = None,
    distance_threshold_col: Optional[float] = None,
    is_percentile: bool = True,
) -> NearestNeighborImputer:
    """Create a two-sided nearest neighbor imputer.

    If distance_threshold_row and distance_threshold_col are not provided, they must be set by calling fit on the imputer.

    Args:
        distance_threshold_row (float): [Optional] Distance threshold for row-row nearest neighbors.
        distance_threshold_col (float): [Optional] Distance threshold for column-column nearest neighbors.
        is_percentile (bool): [Optional] Whether to use percentile-based distance threshold. Defaults to False.

    Returns:
        NearestNeighborImputer: A two-sided nearest neighbor imputer.

    """
    from .estimation_methods import TSEstimator
    from .data_types import Scalar

    estimator = TSEstimator(is_percentile=is_percentile)
    data_type = Scalar()
    if distance_threshold_row is None or distance_threshold_col is None:
        distance_threshold = None
    else:
        # \vec eta = (\eta_row, \eta_col)
        distance_threshold = (distance_threshold_row, distance_threshold_col)

    # distance_threshold = None if (distance_threshold_row is None or distance_threshold_col is None) else (distance_threshold_row, distance_threshold_col) #\vec eta = (\eta_row, \eta_col)
    return NearestNeighborImputer(estimator, data_type, distance_threshold)
