"""Implementation of vanilla NN imputer.

Details of the vanilla NN algorithm can be found in Sec. 2.3 of the following paper:
    Dwivedi, R., Tian, K., Tomkins, S., Klasnja, P., Murphy, S., & Shah, D. (2022).
    Counterfactual inference for sequential experiments. arXiv preprint arXiv:2202.06891.
"""

from .nnimputer import NearestNeighborImputer
from .data_types import Scalar
from .estimation_methods import RowRowEstimator, ColColEstimator
from typing import Optional


def row_row(distance_threshold: Optional[float] = None) -> NearestNeighborImputer:
    """Create a row-row nearest neighbor imputer.

    Args:
        distance_threshold (float, optional): Distance threshold for nearest neighbors. Defaults to None.

    Returns:
        NearestNeighborImputer: A row-row nearest neighbor imputer.

    """
    estimator = RowRowEstimator()
    data_type = Scalar()
    return NearestNeighborImputer(estimator, data_type, distance_threshold)


def col_col(distance_threshold: Optional[float] = None) -> NearestNeighborImputer:
    """Create a column-column nearest neighbor imputer.

    Args:
        distance_threshold (float, optional): Distance threshold for nearest neighbors. Defaults to None.

    Returns:
        NearestNeighborImputer: A column-column nearest neighbor imputer.

    """
    estimator = ColColEstimator()
    data_type = Scalar()
    return NearestNeighborImputer(estimator, data_type, distance_threshold)
