"""Base class for all nearest neighbors algorithms

TODO: resolve ruff errors
"""

import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Optional


class DataType(ABC):
    """Abstract class for data types. Examples include scalars and distributions."""

    @abstractmethod
    def distance(self, obj1, obj2) -> float:
        """Calculate the distance between two objects.

        Args:
            obj1: Object 1
            obj2: Object 2

        """
        pass

    @abstractmethod
    def average(self, object_list: list):
        """Calculate the average of a list of objects.

        Args:
            object_list (list): List of objects

        """
        pass


class EstimationMethod(ABC):
    """Abstract class for estimation methods.
    Examples include user-user (row-wise), item-item (column-wise), two-sided, and doubly-robust.
    """

    @abstractmethod
    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        data_type: DataType,
    ) -> npt.NDArray:
        """Impute the missing value at the given row and column.

        Args:
            row (int): Row index
            column (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        Returns:
            npt.NDArray: Imputed value

        """
        pass


class NearestNeighborImputer:
    """Nearest neighbor composed of different kinds of methods."""

    def __init__(
        self,
        estimation_method: EstimationMethod,
        data_type: DataType,
        distance_treshold: Optional[float] = None,
    ):
        """Initialize the imputer.

        Args:
            estimation_method (EstimationMethod): Estimation method to use (e.g. user-user, item-item, two-sided, doubly-robust)
            data_type (DataType): Data type to use (e.g. scalars, distributions)
            distance_treshold (Optional[float], optional): Distance threshold to use. Defaults to None.

        """
        self.estimation_method = estimation_method
        self.data_type = data_type
        self.distance_threshold = None

    def __str__(self):
        return f"NearestNeighborImputer(estimation_method={self.estimation_method}, data_type={self.data_type})"

    def impute(
        self, row: int, column: int, data_array: npt.NDArray, mask_array: npt.NDArray
    ) -> npt.NDArray:
        """Impute the missing value at the given row and column.

        Args:
            row (int): Row index
            column (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix

        Raises:
            ValueError: If distance threshold is not set

        Returns:
            npt.NDArray: Imputed value

        """
        if self.distance_threshold is None:
            raise ValueError(
                "Distance threshold is not set. Call a FitMethod on this imputer or manually set it."
            )
        return self.estimation_method.impute(
            row, column, data_array, mask_array, self.data_type
        )


class FitMethod(ABC):
    """Abstract class for fiting methods.
    Examples include cross validation methods.
    """

    @abstractmethod
    def fit(
        self,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        imputer: NearestNeighborImputer,
    ) -> float:
        """Find the best distance threshold for the given data.

        Args:
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            imputer (NearestNeighborImputer): Imputer object

        Returns:
            float: Best distance threshold
        """
        pass
