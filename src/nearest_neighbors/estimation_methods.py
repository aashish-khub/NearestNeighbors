from .nnimputer import EstimationMethod, DataType
import numpy.typing as npt
import numpy as np


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
        nearest_neighbors_data = data_array[nearest_neighbors, column].flatten()
        
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
        
        return RowRowEstimator().impute(column, row, data_transposed, mask_transposed, distance_threshold, data_type)
