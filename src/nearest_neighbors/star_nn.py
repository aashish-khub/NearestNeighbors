"""A function to return the Star NN imputer.
"""

from typing import Optional
import numpy.typing as npt
import numpy as np
# from nearest_neighbors.weighted_estimation_methods import StarNNEstimator
from nearest_neighbors.data_types import Scalar
from nearest_neighbors.nnimputer import DataType


class StarNNEstimator:
    """Estimate the missing value using Star NN.
    """

    def __str__(self):
        return "StarNNEstimator"
    

    def impute(
        self,
        row: int,
        column: int,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        data_type: DataType,
        noise_variance: float,
        row_distances: npt.NDArray,
        delta: float = 0.05
    ) -> npt.NDArray:
        """Impute the missing value at the given row and column using Star NN.
        """
        observed_rows = np.where(mask_array[:, column] == 1)[0]
        n_observed = len(observed_rows)
        
        if n_observed == 0:
            return np.array(np.nan)
        
        
        row_distances_copy = row_distances[np.ix_(observed_rows, observed_rows)] - 2 * noise_variance
        np.fill_diagonal(row_distances_copy, 0)
        
        mean_distance = np.mean(row_distances_copy[0, :])
        dist_diff = row_distances_copy[0, :] - mean_distance
        
        weights = (1/n_observed) - dist_diff/(8 * noise_variance * np.log(1/delta))
        
        sorted_weights = np.sort(weights)[::-1]
        weight_sum = 0
        u = 0
        
        for k in range(n_observed):
            weight_sum += sorted_weights[k]
            u_new = (weight_sum - 1) / (k + 1)
            if k == n_observed - 1 or sorted_weights[k + 1] <= u_new:
                u = u_new
                break
        
        weights = np.maximum(0, weights - u)
        # print("weights for row %d, column %d: %s" % (row, column, weights))
        
        return np.sum(weights * data_array[observed_rows, column]) 



class star_nn:
    
    def __init__(
        self,
        noise_variance: float = None,
        delta: float = 0.05,
        row_distances: Optional[npt.NDArray] = None,
    ):
        
        self.data_type = Scalar()
        self.estimation_method = StarNNEstimator()
        self.noise_variance = noise_variance
        self.delta = delta
        self.row_distances = row_distances
        
    def __str__(self):
        return "StarNNImputer(noise_variance={}, delta={})".format(self.noise_variance, self.delta)
    
    def set_row_distances(
        self, 
        data_array: Optional[npt.NDArray] = None, 
        mask_array: Optional[npt.NDArray] = None,
        distance_matrix: Optional[npt.NDArray] = None
    ) -> None:
        """Set the row distances for the imputer.
        
        This method can either:
        1. Calculate row distances from data_array and mask_array, or
        2. Validate and set a pre-computed distance matrix
        
        Args:
            data_array (Optional[npt.NDArray], optional): Data matrix. Required if distance_matrix is None. Defaults to None.
            mask_array (Optional[npt.NDArray], optional): Mask matrix. Required if distance_matrix is None. Defaults to None.
            distance_matrix (Optional[npt.NDArray], optional): Pre-computed distance matrix. Defaults to None.
            
        Raises:
            ValueError: If neither data_array/mask_array nor distance_matrix is provided
            ValueError: If distance_matrix is not a square matrix
            ValueError: If distance_matrix contains negative values
            ValueError: If distance_matrix is not symmetric
            ValueError: If distance_matrix diagonal is not all zeros
        """
        if distance_matrix is not None:
            # Validate the provided distance matrix
            if not isinstance(distance_matrix, np.ndarray):
                raise ValueError("distance_matrix must be a numpy array")
                
            if len(distance_matrix.shape) != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
                raise ValueError("distance_matrix must be a square matrix")
                
            if np.any(distance_matrix < 0):
                raise ValueError("distance_matrix cannot contain negative values")
                
            if not np.allclose(distance_matrix, distance_matrix.T):
                raise ValueError("distance_matrix must be symmetric")
                
            if not np.allclose(np.diag(distance_matrix), 0):
                raise ValueError("distance_matrix diagonal must be all zeros")
                
            self.row_distances = distance_matrix
        elif data_array is not None and mask_array is not None:
            # Calculate row distances from data
            n_rows, n_cols = data_array.shape
            self.row_distances = np.zeros((n_rows, n_rows))
            for i in range(n_rows):
                for j in range(i+1, n_rows):
                    overlap_cols = np.logical_and(mask_array[i, :], mask_array[j, :])
                    if not np.any(overlap_cols):
                        self.row_distances[i, j] = np.inf
                        self.row_distances[j, i] = np.inf
                        continue
                    for k in range(n_cols):
                        if not overlap_cols[k]:
                            continue
                        self.row_distances[i, j] += self.data_type.distance(data_array[i, k], data_array[j, k])
                    self.row_distances[i, j] /= np.sum(overlap_cols)
                    self.row_distances[j, i] = self.row_distances[i, j]
        else:
            raise ValueError("Either distance_matrix or both data_array and mask_array must be provided")

    def set_noise_variance(self, noise_variance: float) -> None:
        self.noise_variance = noise_variance
    
    def impute(self, row: int, column: int, data_array: npt.NDArray, mask_array: npt.NDArray) -> float:
        return self.estimation_method.impute(row, column, data_array, mask_array, self.data_type, self.noise_variance, self.row_distances, self.delta)
    
    def impute_matrix(self, data_array: npt.NDArray, mask_array: npt.NDArray) -> float:
        n_rows, n_cols = data_array.shape
        imputed_data = np.zeros_like(data_array)
        for i in range(n_rows):
            for j in range(n_cols):
                imputed_data[i, j] = self.estimation_method.impute(i, j, data_array, mask_array, self.data_type, self.noise_variance, self.row_distances, self.delta)
        return imputed_data
    
    def fit(self, 
            data_array: npt.NDArray,
            mask_array: npt.NDArray, 
            max_iterations: int = 10,
            convergence_threshold: float = 1e-4,
            ) -> None:
        
        n_rows, n_cols = data_array.shape
        
        self.delta = 0.5/np.sqrt(n_rows)
        
        if self.row_distances is None:
            self.set_row_distances(data_array, mask_array)
        if self.noise_variance is None:
            self.set_noise_variance(np.var(data_array[mask_array == 1])/2)
        
        imputed_data = np.zeros_like(data_array)
        
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}/{max_iterations} with noise variance: {self.noise_variance}")
            
            for i in range(n_rows):
                for j in range(n_cols):
                    imputed_data[i, j] = self.estimation_method.impute(i, j, data_array, mask_array, self.data_type, self.noise_variance, self.row_distances, self.delta)
            
            diff = imputed_data[mask_array == 1] - data_array[mask_array == 1]
            diff = diff[~np.isnan(diff)]  # Remove any NaN values
            if len(diff) > 0:
                new_variance_estimate = np.var(diff)
                if abs(new_variance_estimate - self.noise_variance) / self.noise_variance < convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations and final noise variance: {new_variance_estimate}")
                    self.noise_variance = new_variance_estimate
                    break
                self.noise_variance = new_variance_estimate
        
        return imputed_data
            
            
             
            
            
