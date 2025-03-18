from .nnimputer import FitMethod, DataType, NearestNeighborImputer
import numpy.typing as npt
import numpy as np
from hyperopt import hp, fmin, tpe

def evaluate_imputation(data_array: npt.NDArray, 
                        mask_array: npt.NDArray, 
                        imputer: NearestNeighborImputer,
                        test_cells: list[tuple[int,int]],
                        data_type: DataType) -> float:
    """Evaluate the imputer on a set.
    
    Args:
        data_array (npt.NDArray): Data matrix
        mask_array (npt.NDArray): Mask matrix
        imputer (NearestNeighborImputer): Imputer object
        test_cells (list[tuple[int,int]]): List of cells as tuples of row and column indices
        data_type (DataType): Data type to use (e.g. scalars, distributions)
            
    Raises:
        ValueError: If a validation cell is missing
        
    Returns:
        float: Average imputation error
        
    """
    error = 0
    for row, col in test_cells:
        if mask_array[row,col] == 0:
            raise ValueError("Validation cell is missing.")
        mask_array[row,col] = 0 # Set the mask to 0
        imputed_value = imputer.impute(row, col, data_array, mask_array)
        true_value = data_array[row,col]
        
        error += data_type.distance(imputed_value, true_value)
        
        mask_array[row,col] = 1 # Reset the mask
        
    return error / len(test_cells)

class KFoldValidation(FitMethod):
    """Fit method for k-fold cross validation for just one row."""
    
    def __init__(self, k: int):
        """Initialize the k-fold cross validation method.
        
        Args:
            k (int): Number of folds
            
        """
        self.k = k
        
    def fit(
        self,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        imputer: NearestNeighborImputer,
    ) -> float:
        """Find the best distance threshold for the given data
        using k-fold cross validation.

        Args:
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            imputer (NearestNeighborImputer): Imputer object

        Returns:
            float: Best distance threshold
            
        """
        pass