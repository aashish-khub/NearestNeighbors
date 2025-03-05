"""Implementation of vanilla nearest neighbor imputation."""

import numpy as np
import numpy.typing as npt

from .nnimputer import DataType, EstimationMethod


class VectorDataType(DataType):
    def __init__(self):
        super().__init__()

    def distance(self, obj1: npt.NDArray, obj2: npt.NDArray) -> float:
        """Compute the Euclidean distance between two vectors.

        Args:
            obj1 (npt.NDArray): The first vector.
            obj2 (npt.NDArray): The second vector.

        Returns:
            float: The Euclidean distance between the two vectors.

        """
        # assert that x is a vector
        assert obj1.ndim == 1
        # assert that y is a vector
        assert obj2.ndim == 1
        return float(np.linalg.norm(obj1 - obj2))

    def average(self, object_list: npt.NDArray) -> float:
        """Compute the average of a vector.

        Args:
            object_list (npt.NDArray): A list of vectors.

        Returns:
            float: The average of the vector.

        """
        assert object_list.ndim == 2
        return np.mean(object_list, axis=0)


class RowRowEstimator(EstimationMethod):
    pass


class ColumnColumnEstimator(RowRowEstimator):
    # TODO: this is the same as RowRowEstimator
    # except the matrix is transposed
    pass
