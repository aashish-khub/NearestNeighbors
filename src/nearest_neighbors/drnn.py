"""A function to return the DRNN imputer."""
from .nnimputer import NearestNeighborImputer
from typing import Optional
import numpy as np

def drnn(
    distance_threshold_row: Optional[float] = None,
    distance_threshold_col: Optional[float] = None
) -> NearestNeighborImputer:
    """Create a doubly robust nearest neighbor imputer.

    If distance_threshold_row and distance_threshold_col are not provided, they must be set by calling fit on the imputer.

    Args:
        distance_threshold_row (float): [Optional] Distance threshold for row-row nearest neighbors.
        distance_threshold_col (float): [Optional] Distance threshold for column-column nearest neighbors.

    Returns:
        NearestNeighborImputer: A doubly robust nearest neighbor imputer.

    """
    from .estimation_methods import DREstimator
    from .data_types import Scalar

    estimator = DREstimator(distance_threshold_row, distance_threshold_col)
    data_type = Scalar()
    # note that the default value of distance_threshold is np.inf -> distance threshold is unused for DRNN
    return NearestNeighborImputer(estimator, data_type, np.inf)