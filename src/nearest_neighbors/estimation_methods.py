from .nnimputer import EstimationMethod, DataType
import numpy.typing as npt
import numpy as np
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)


class RowRowEstimator(EstimationMethod):
    """Estimate the missing value using row-row nearest neighbors."""

    def __init__(self):
        self.row_distances = dict()

    def __str__(self):
        return "RowRowEstimator"

    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        distance_threshold: Union[float, Tuple[float, float]],
        data_type: DataType,
    ) -> npt.NDArray:
        """Impute the missing value at the given row and column.

        Args:
            row (int): Row index
            column (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            distance_threshold (float): Distance threshold for nearest neighbors
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        Returns:
            npt.NDArray: Imputed value

        """
        # Calculate distances between rows
        self._calculate_distances(row, column, data_array, mask_array, data_type)
        # Find the nearest neighbors indexes
        nearest_neighbors = np.where(self.row_distances[row] <= distance_threshold)[0]
        # Apply mask_array to data_array
        masked_data_array = np.where(mask_array, data_array, np.nan)

        # If no neighbors found, return nan
        if len(nearest_neighbors) == 0:
            # return np.array(np.nan)
            # NOTE: implement the base case described by Eq. 11 in
            # "Counterfactual Inference for Sequential Experiments".
            if mask_array[row, column]:
                # return the observed outcome
                return data_array[row, column]
            else:
                # return the average of all observed outcomes corresponding
                # to treatment 1 at time t.
                return np.array(np.nanmean(masked_data_array[:, column]))

        # Calculate the average of the nearest neighbors
        nearest_neighbors_data = masked_data_array[nearest_neighbors, column]

        return data_type.average(nearest_neighbors_data)

    def _calculate_distances(
        self,
        row: int,
        col: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        data_type: DataType,
    ) -> None:
        """Sets the distances for the imputer.
        Sets the distances as a class attribute, so returns nothing.

        Args:
            row (int): Row index
            col (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        """
        data_shape = data_array.shape
        n_rows = data_shape[0]
        n_cols = data_shape[1]

        if row in self.row_distances:
            return

        # Calculate distances between rows
        row_dists = np.zeros(n_rows)
        for i in range(n_rows):
            # Get columns observed in both row i and row
            overlap_columns = np.logical_and(mask_array[row], mask_array[i])

            if not np.any(overlap_columns):
                row_dists[i] = np.inf
                continue

            # Calculate distance between rows
            for j in range(n_cols):
                if (
                    not overlap_columns[j] or j == col
                ):  # Skip missing values and the target column
                    continue
                row_dists[i] += data_type.distance(data_array[row, j], data_array[i, j])
            row_dists[i] /= np.sum(overlap_columns)
        self.row_distances[row] = row_dists


class ColColEstimator(EstimationMethod):
    """Estimate the missing value using column-column nearest neighbors."""

    def __init__(self):
        self.estimator = RowRowEstimator()
        # use the same logic as RowRowEstimator but transposed
        # save the distances

    def __str__(self):
        return "ColColEstimator"

    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        distance_threshold: Union[float, Tuple[float, float]],
        data_type: DataType,
    ) -> npt.NDArray:
        """Impute the missing value at the given row and column.

        Args:
            row (int): Row index
            column (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            distance_threshold (float): Distance threshold for nearest neighbors
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        Returns:
            npt.NDArray: Imputed value

        """
        data_transposed = np.swapaxes(data_array, 0, 1)
        mask_transposed = np.swapaxes(mask_array, 0, 1)

        return self.estimator.impute(
            column, row, data_transposed, mask_transposed, distance_threshold, data_type
        )

    def _calculate_distances(
        self,
        row: int,
        col: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        data_type: DataType,
    ) -> None:
        """Sets the distances for the imputer.
        Sets the distances as a class attribute, so returns nothing.

        Args:
            row (int): Row index
            col (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        """
        # We use the RowRow Estimator to calculate the distances
        pass


class DREstimator(EstimationMethod):
    """Estimate the missing entry using doubly robust nearest neighbors."""

    def __init__(self):
        self.row_distances = dict()
        self.col_distances = dict()

    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        distance_threshold: Union[float, Tuple[float, float]],
        data_type: DataType,
    ) -> npt.NDArray:
        """Impute the missing value at the given row and column using doubly robust method.

        Args:
        ----
            row (int): Row index
            column (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            distance_threshold (float or Tuple[float, float]): Distance threshold for nearest neighbors
            or a tuple of (row_threshold, col_threshold) for row and column respectively.
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        """
        if isinstance(distance_threshold, tuple):
            distance_threshold_row = distance_threshold[0]
            distance_threshold_col = distance_threshold[1]
        else:
            distance_threshold_row = distance_threshold
            distance_threshold_col = distance_threshold

        self._calculate_distances(row, column, data_array, mask_array, data_type)
        # Find the row nearest neighbors indexes
        row_nearest_neighbors = np.nonzero(
            self.row_distances[row] <= distance_threshold_row
        )[0]

        # Find the col nearest neighbors indexes
        col_nearest_neighbors = np.nonzero(
            self.col_distances[column] <= distance_threshold_col
        )[0]

        # neighbors can only be used if they are observed
        row_nearest_neighbors = row_nearest_neighbors[
            mask_array[row_nearest_neighbors, column] == 1
        ]
        col_nearest_neighbors = col_nearest_neighbors[
            mask_array[row, col_nearest_neighbors] == 1
        ]

        # Use doubly robust nearest neighbors to combine row and col
        y_itprime = data_array[row, col_nearest_neighbors]
        y_jt = data_array[row_nearest_neighbors, column]
        if len(y_itprime) == 0 and len(y_jt) == 0:
            return np.array(np.nan)

        # get intersecting entries
        j_inds, tprime_inds = np.meshgrid(
            row_nearest_neighbors, col_nearest_neighbors, indexing="ij"
        )
        y_jtprime = data_array[j_inds, tprime_inds]
        mask_jtprime = mask_array[j_inds, tprime_inds]

        # nonzero gets all indices of the mask that are 1
        intersec_inds = np.nonzero(mask_jtprime == 1)

        y_itprime_inter = y_itprime[
            intersec_inds[1]
        ]  # this is a array of column values for all intersecting triplets
        y_jt_inter = y_jt[
            intersec_inds[0]
        ]  # this is a array of row values for all intersecting triplets
        # note: defaults to rownn if no intersection -> should default to ts-nn instead?
        if len(y_itprime_inter) == 0 or len(y_jt_inter) == 0:
            return np.array(
                data_type.average(y_jt)
                if len(y_jt) > 0
                else data_type.average(y_itprime)
            )
        sum_y = y_itprime_inter + y_jt_inter - y_jtprime[intersec_inds]
        avg = data_type.average(sum_y)
        return avg

    def _calculate_distances(
        self,
        row: int,
        col: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        data_type: DataType,
    ) -> None:
        """Sets the distances for the imputer.
        Sets the distances as a class attribute, so returns nothing.

        Args:
            row (int): Row index
            col (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        """
        data_shape = data_array.shape
        n_rows = data_shape[0]
        n_cols = data_shape[1]

        if row not in self.row_distances:
            row_dists = np.zeros(n_rows)
            for i in range(n_rows):
                # Get columns observed in both row i and row
                overlap_columns = np.logical_and(mask_array[row], mask_array[i])

                if not np.any(overlap_columns):
                    row_dists[i] = np.inf
                    continue

                # Calculate distance between rows
                for j in range(n_cols):
                    if (
                        not overlap_columns[j] or j == col
                    ):  # Skip missing values and the target column
                        continue
                    row_dists[i] += data_type.distance(
                        data_array[row, j], data_array[i, j]
                    )
                row_dists[i] /= np.sum(overlap_columns)
            self.row_distances[row] = row_dists
            # self.row_distances[row][row] = np.inf  # Exclude the row itself

        if col not in self.col_distances:
            col_dists = np.zeros(n_cols)
            for i in range(n_cols):
                # Get rows observed in both column i and column
                overlap_rows = np.logical_and(mask_array[:, col], mask_array[:, i])

                if not np.any(overlap_rows):
                    col_dists[i] = np.inf
                    continue

                # Calculate distance between columns
                for j in range(n_rows):
                    if not overlap_rows[j] or j == row:
                        continue
                    col_dists[i] += data_type.distance(
                        data_array[j, col], data_array[j, i]
                    )
                col_dists[i] /= np.sum(overlap_rows)
            self.col_distances[col] = col_dists
            # self.col_distances[col][col] = np.inf


class TSEstimator(EstimationMethod):
    """Estimate the missing value using two-sided nearest neighbors.

    This method first finds the row and column neighborhoods (based on a
    distance computed over overlapping observed entries, excluding the target)
    and then imputes the missing entry by averaging the observed values
    over the cross-product of these neighborhoods.
    """

    def __init__(self):
        self.row_distances = dict()
        self.col_distances = dict()

    def __str__(self):
        return "TSEstimator"

    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        distance_threshold: Union[float, Tuple[float, float]],
        data_type: DataType,
    ) -> npt.NDArray:
        r"""Impute the missing value at the given row and column using two-sided NN.

        Args:
            row (int): Target row index.
            column (int): Target column index.
            data_array (npt.NDArray): Data matrix.
            mask_array (npt.NDArray): Boolean mask matrix (True if observed).
            distance_threshold (float or Tuple[float, float]): Distance threshold(s) for row and column neighborhoods. This is our \vec eta = (\eta_row, \eta_col).
            data_type (DataType): Provides methods for computing distances and averaging.

        Returns:
            npt.NDArray: Imputed value.

        """
        if isinstance(distance_threshold, tuple):
            eta_row, eta_col = distance_threshold
        else:
            eta_row = distance_threshold
            eta_col = distance_threshold

        self._calculate_distances(row, column, data_array, mask_array, data_type)
        # Establish the neighborhoods subject to the distance thresholds
        row_nearest_neighbors = np.where(self.row_distances[row] <= eta_row)[
            0
        ]  # This is the set N_row(i, j) = {i' | d^2(i, i') <= eta_row^2}
        col_nearest_neighbors = np.where(self.col_distances[column] <= eta_col)[
            0
        ]  # This is the set N_col(i, j) = {j' | d^2(j, j') <= eta_col^2}
        neighborhood_submatrix = data_array[
            np.ix_(row_nearest_neighbors, col_nearest_neighbors)
        ]  # This is the submatrix of the cross-product of the row and column neighborhoods
        mask_array = mask_array.astype(bool)  # for efficient indexing
        neighborhood_mask = mask_array[
            np.ix_(row_nearest_neighbors, col_nearest_neighbors)
        ]  # This is the mask of the cross-product of the row and column neighborhoods

        values_for_estimation = neighborhood_submatrix[neighborhood_mask.astype(bool)]
        if values_for_estimation.size == 0:
            if mask_array[row, column]:
                logger.log(
                    logging.WARNING,
                    f"Warning: No valid neighbors found for ({row}, {column}). Returning observed value.",
                )
                return data_array[row, column]
            else:
                logger.log(
                    logging.WARNING,
                    f"Warning: No valid neighbors found for ({row}, {column}). Returning np.nan.",
                )
                return np.array(np.nan)
        else:
            theta_hat = data_type.average(values_for_estimation)
            return theta_hat

    def _calculate_distances(
        self,
        row: int,
        col: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        data_type: DataType,
    ) -> None:
        """Sets the distances for the imputer.
        Sets the distances as a class attribute, so returns nothing.

        Args:
            row (int): Row index
            col (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        """
        n_rows, n_cols = data_array.shape

        if row not in self.row_distances:
            row_dists = np.zeros(n_rows)
            for i in range(n_rows):
                # Get columns observed in both row i and row
                overlap_columns = np.logical_and(mask_array[row], mask_array[i])

                if not np.any(overlap_columns):
                    row_dists[i] = np.inf
                    continue

                # Calculate distance between rows
                for j in range(n_cols):
                    if (
                        not overlap_columns[j] or j == col
                    ):  # Skip missing values and the target column
                        continue
                    row_dists[i] += data_type.distance(
                        data_array[row, j], data_array[i, j]
                    )
                row_dists[i] /= np.sum(overlap_columns)
            row_dists[row] = np.inf  # Exclude the row itself
            self.row_distances[row] = row_dists

        if col not in self.col_distances:
            col_dists = np.zeros(n_cols)
            for i in range(n_cols):
                # Get rows observed in both column i and column
                overlap_rows = np.logical_and(mask_array[:, col], mask_array[:, i])

                if not np.any(overlap_rows):
                    col_dists[i] = np.inf
                    continue

                # Calculate distance between columns
                for j in range(n_rows):
                    if not overlap_rows[j] or j == row:
                        continue
                    col_dists[i] += data_type.distance(
                        data_array[j, col], data_array[j, i]
                    )
                col_dists[i] /= np.sum(overlap_rows)
            col_dists[col] = np.inf
            self.col_distances[col] = col_dists
