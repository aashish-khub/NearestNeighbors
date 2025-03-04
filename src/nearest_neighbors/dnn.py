# Base classes for distributional nearest neighbors

import numpy as np
import numpy.typing as npt


from collections import defaultdict

from abc import ABC, abstractmethod


class DistNNEstimator(ABC):
    """Abstract base class for distributional nearest neighbors estimators"""

    def __init__(self, nn_type="user-user", rand_seed=None):
        """Initializes the distributional nearest neighbors estimator."""
        if nn_type not in ["user-user", "item-item"]:
            raise ValueError('nn_type must be one of "user-user" or "item-item".')

        self.nn_type = nn_type
        self.rand_seed = rand_seed
        self.distance_threshold = None

    @abstractmethod
    def distributional_distance(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> float:
        """Returns a distance between two empirical distributions of vectors.

        Args:
            x (npt.NDArray[np.float64]): first empirical distribution
            y (npt.NDArray[np.float64]): second empirical distribution

        Returns:
            float: distance between the two empirical distributions

        """
        pass

    @abstractmethod
    def average_distributions(
        self, distributions: list[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Returns the average of the empirical distributions.

        Args:
            distributions (list[npt.NDArray[np.float64]]): list of empirical distributions

        Returns:
            npt.NDArray[np.float64]: average empirical distribution

        """
        pass

    @abstractmethod
    def fit(self, data: list[list[npt.NDArray[np.float64]]], distance_threshold: float):
        """Fits the distributional nearest neighbors estimator to the data.

        Args:
            data (list[list[npt.NDArray[np.float64]]]): 2-dimensional nested list of data arrays.
            distance_threshold (float): distance threshold for nearest neighbors

        """
        pass

    def predict(
        self,
        row_index: int,
        column_index: int,
        data: list[list[npt.NDArray[np.float64]]],
    ) -> npt.NDArray[np.float64]:
        """Predicts the value of the cell at the specified row and column index.
        The data is a 2-dimensional nested list of data arrays.
        The outer list represents the rows and the inner list represents the columns.
        None or np.NaN values represent missing data.

        Args:
            row_index (int): row index to impute
            column_index (int): column index to impute
            data (list[list[npt.NDArray[np.float64]]]): 2-dimensional nested list of data arrays.

        Raises:
            ValueError: _description_

        Returns:
            npt.NDArray[np.float64]: _description_

        """
        if self.distance_threshold is None:
            raise ValueError("Distance threshold is not set. Call fit() first.")

        average_cells = None
        distance_cells = None

        num_rows = len(data)
        num_columns = len(data[0])

        if self.nn_type == "user-user":
            # Average over the column of interest
            average_cells = [(i, column_index) for i in range(num_rows)]

            # Exclude current column when calculating distances
            distance_cells = [
                (row_index, j) for j in range(num_columns) if j != column_index
            ]

        elif self.nn_type == "item-item":
            # Average over the row of interest
            average_cells = [(row_index, j) for j in range(num_columns)]

            # Exclude current row when calculating distances
            distance_cells = [
                (i, column_index) for i in range(num_rows) if i != row_index
            ]

        # Calculate the average distance between rows or columns (depending on nn_type)
        distance_dict = defaultdict(float)

        for i, j in distance_cells:
            if data[i][j] is None:
                continue
            other_cells = None
            key = None
            if self.nn_type == "user-user":
                other_cells = [(k, j) for k in range(num_rows)]
            elif self.nn_type == "item-item":
                other_cells = [(i, k) for k in range(num_columns)]
            # Calculate the distances for each cell in distance_cells
            for k, l in other_cells:
                key = k if self.nn_type == "user-user" else l
                # Add to total distance for the row or column
                if data[k][l] is not None:
                    distance_dict[key] += self.distributional_distance(
                        data[i][j], data[k][l]
                    )

        # Calculate the average distance for each row or column
        for key in distance_dict.keys():
            distance_dict[key] /= len(distance_cells)

        # Find the nearest neighbors
        nearest_neighbors = []
        if self.nn_type == "user-user":
            for i, dist in distance_dict.items():
                if dist < self.distance_threshold and data[i][column_index] is not None:
                    nearest_neighbors.append((i, column_index))
        elif self.nn_type == "item-item":
            for j, dist in distance_dict.items():
                if dist < self.distance_threshold and data[row_index][j] is not None:
                    nearest_neighbors.append((row_index, j))

        neighbor_distributions = [data[i][j] for i, j in nearest_neighbors]

        return self.average_distributions(neighbor_distributions)
