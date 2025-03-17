from .nnimputer import EstimationMethod, DataType
import numpy.typing as npt
import numpy as np
from typing import Optional

class RowRowEstimator(EstimationMethod):
    """Estimate the missing value using row-row nearest neighbors."""

    def __str__(self):
        return "RowRowEstimator"

    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        distance_threshold: float,
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
        data_shape = data_array.shape
        n_rows = data_shape[0]
        n_cols = data_shape[1]

        # Calculate distances between rows
        row_distances = np.zeros(n_rows)
        for i in range(n_rows):
            # Get columns observed in both row i and row
            overlap_columns = np.logical_and(mask_array[row], mask_array[i])

            if not np.any(overlap_columns):
                row_distances[i] = np.inf
                continue

            # Calculate distance between rows
            for j in range(n_cols):
                if (
                    not overlap_columns[j] or j == column
                ):  # Skip missing values and the target column
                    continue
                row_distances[i] += data_type.distance(
                    data_array[row, j], data_array[i, j]
                )
            row_distances[i] /= np.sum(overlap_columns)

        # Find the nearest neighbors indexes
        nearest_neighbors = np.where(row_distances <= distance_threshold)[0]

        # If no neighbors found, return nan
        if len(nearest_neighbors) == 0:
            return np.array(np.nan)

        # Calculate the average of the nearest neighbors
        nearest_neighbors_data = data_array[nearest_neighbors, column]

        return data_type.average(nearest_neighbors_data)


class ColColEstimator(EstimationMethod):
    """Estimate the missing value using column-column nearest neighbors."""

    def __str__(self):
        return "ColColEstimator"

    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        distance_threshold: float,
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

        return RowRowEstimator().impute(
            column, row, data_transposed, mask_transposed, distance_threshold, data_type
        )

class DREstimator(EstimationMethod):
    """Estimate the missing entry using doubly robust nearest neighbors."""
   
    def __init__(self,
        distance_threshold_row: Optional[float] = None,
        distance_threshold_col: Optional[float] = None
    ) -> None:
        super().__init__()
        self.distance_threshold_row = distance_threshold_row
        self.distance_threshold_col = distance_threshold_col

    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        distance_threshold: float,
        data_type: DataType,
    ) -> npt.NDArray:
        """Impute the missing value at the given row and column using doubly robust method.
        
        Args:
        ----
            row (int): Row index
            column (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            distance_threshold (float): UNUSED for DRNN -> see __init__ of DREstimator for proper specification of distance thresholds
            data_type (DataType): Data type to use (e.g. scalars, distributions)
        
        """
        if self.distance_threshold_row is None or self.distance_threshold_col is None:
            raise ValueError("Distance thresholds for row and column must be set for DREstimator. Please call fit on the imputer first or provide the thresholds directly to the drnn method.")
        data_shape = data_array.shape
        n_rows = data_shape[0]
        n_cols = data_shape[1]

        row_distances = np.zeros(n_rows)
        for i in range(n_rows):
            # Get columns observed in both row i and row
            overlap_columns = np.logical_and(mask_array[row], mask_array[i])

            if not np.any(overlap_columns):
                row_distances[i] = np.inf
                continue

            # Calculate distance between rows
            for j in range(n_cols):
                if (
                    not overlap_columns[j] or j == column
                ):  # Skip missing values and the target column
                    continue
                row_distances[i] += data_type.distance(
                    data_array[row, j], data_array[i, j]
                )
            row_distances[i] /= np.sum(overlap_columns)

        # Find the row nearest neighbors indexes
        row_nearest_neighbors = np.where(row_distances <= self.distance_threshold_row)[0]

        col_distances = np.zeros(n_cols)
        for i in range(n_cols):
            # Get rows observed in both column i and column
            overlap_rows = np.logical_and(mask_array[:, column], mask_array[:, i])

            if not np.any(overlap_rows):
                col_distances[i] = np.inf
                continue

            # Calculate distance between columns
            for j in range(n_rows):
                if (
                    not overlap_rows[j] or j == row
                ):
                    continue
                col_distances[i] += data_type.distance(
                    data_array[j, column], data_array[j, i]
                )
            col_distances[i] /= np.sum(overlap_rows)

        # Find the col nearest neighbors indexes
        col_nearest_neighbors = np.where(col_distances <= self.distance_threshold_col)[0]

        # Use doubly robust nearest neighbors to combine row and col
        y_itprime = data_array[row, col_nearest_neighbors]
        y_jt = data_array[row_nearest_neighbors, column]

        if len(y_itprime) == 0 and len(y_jt) == 0:
            return np.array(np.nan)

        # get intersecting entries
        j_grid, tprime_grid = np.meshgrid(row_nearest_neighbors, col_nearest_neighbors, indexing='ij')
        y_jtprime = data_array[j_grid, tprime_grid]
        mask_jtprime = mask_array[j_grid, tprime_grid]

        intersec_inds = np.nonzero(mask_jtprime == 1)

        y_itprime_inter = y_itprime[intersec_inds]
        y_jt_inter = y_jt[intersec_inds]
        # note: defaults to rownn if no intersection -> should default to ts-nn instead?
        if len(y_itprime_inter) == 0 or len(y_jt_inter) == 0:
            return data_type.average(y_jt) if len(y_jt) > 0 else np.average(y_itprime)

        sum_y = y_itprime_inter[np.newaxis, :] + y_jt_inter[:, np.newaxis] - y_jtprime[intersec_inds]
        avg = data_type.average(sum_y)      
        return avg
