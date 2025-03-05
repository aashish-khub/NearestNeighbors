from .nnimputer import DataType
import numpy.typing as npt
from typing import Any
import numpy as np


class Scalar(DataType):
    """Data type for scalars."""

    def distance(self, obj1: float, obj2: float) -> float:
        """Calculate the distance between two scalars.

        Args:
            obj1 (float): Scalar 1
            obj2 (float): Scalar 2

        Returns:
            float: Distance between the two scalars

        """
        return (obj1 - obj2) ** 2

    def average(self, object_list: npt.NDArray[Any]) -> Any:
        """Calculate the average of a list of scalars.

        Args:
            object_list (list[Any]): List of scalars

        Returns:
            Any: Average of the scalars

        """
        return np.mean(object_list)
