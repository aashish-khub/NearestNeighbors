import numpy as np
import numpy.typing as npt

from .nnimputer import DataType
from .nnimputer import EstimationMethod
from .data_types import DistributionKernelMMD

kernel = "linear"

nn_op = DistributionKernelMMD(kernel) # distance and average functions

class RowRowDistNN(EstimationMethod):
    def __init__(self):
        super().__init__()

    def obs_Overlap(row_i: int, row_j: int, mask_array: npt.NDArray) -> list[int]:
        """ Overlapping column indices for ith and jth row
        Args: 
          row_i, row_j: Row index i and j
          mask_array: Masking matrix 

        Returns:
          Vector with val 1 if overlapping and val 0 otherwise
        """
        T = mask_array.shape[1]
        overlap = []
        for t in range(T) : 
            if mask_array[row_i, t] == 1 and mask_array[row_j, t] == 1:
                overlap.append(t)
            
        return overlap
    
    
    def row_Metric(row_i: int, row_j: int, t: int, data_array: npt.NDArray, mask_array: npt.NDArray, exc_opt: bool) -> float : 
            """ Computes squared MMD based metric between rows of Distributional Matrix with missingness
            Args:
                row_i, row_j : two row indices under comparison
                t : column of interest - so need to exclude this column
                data_array : (N * T * n * d) array 
                mask_array : (N * T) sized matrix indicating which entries are observed(val = 1)
                exc_opt : if True, omit t th column when constructing row metric, if False, include t th column
                
            Returns:
                Metric between row i, j when the target parameter is on the t th column
            """
            
            overlap = RowRowDistNN.obs_Overlap(row_i, row_j, mask_array)

            if exc_opt == True : 
                if sum(np.isin(overlap, t)) == 1 : 
                    overlap = np.delete(overlap, np.where(overlap == t)[0])
            if len(overlap) == 0 : 
                val = 10**5
                return val

            Data_i = data_array[row_i, :, :, :]
            Data_j = data_array[row_j, :, :, :]

            pre_val = np.zeros(len(overlap))
            for tau in range(len(overlap)) :
                tau_ind = overlap[tau]
                pre_val[tau] = nn_op.distance(Data_i[tau_ind, :, :], Data_j[tau_ind, :, :])
            
            val = sum(pre_val)/len(overlap)

            return(val)


    def impute(self, row: int, column: int, data_array: npt.NDArray, mask_array: npt.NDArray, distance_threshold: float, data_type: DataType) -> npt.NDArray:
        """Impute the missing value at (row, column) using the distance threshold and data type.

        Args:
            row (int): The row index of the missing value.
            column (int): The column index of the missing value.
            data_array (npt.NDArray): The data array.
            mask_array (npt.NDArray): The mask array.
            distance_threshold (float): The distance threshold.
            data_type (DataType): The data type.

        Returns:
            npt.NDArray: The imputed distribution (mixture of vectors).
        """
        N, T, n, d = data_array.shape[0], data_array.shape[1], data_array.shape[2], data_array.shape[3]

        row_Dissim_mat = np.zeros( (N, N) )

        # t_0 = 1 # Arbitrary column index when constructing rho_{i, j} under CV process
        for i in range(N - 1) :
            for j in range((i + 1), N) :
                row_Dissim_mat[i, j] = RowRowDistNN.row_Metric(i, j, t, data_array, mask_array, exc_opt = True)
    
        row_Dissim_mat = row_Dissim_mat + np.transpose(row_Dissim_mat)
        row_Dissim_vec = row_Dissim_mat[row, :]

        neighbor_candidate = data_array[:, column, :, :]

        neighbor_ind = np.where( (row_Dissim_vec < distance_threshold)*(mask_array[:, column]) == 1 )[0]

        if sum(np.isin(neighbor_ind, row)) == 1 : # Pretending as IF (i, t) entry is missing
            neighbor_ind = np.delete(neighbor_ind, np.where(neighbor_ind == row)[0])

        if len(neighbor_ind) == 0 :
            neighbor = np.zeros( (n, d) )
            return neighbor

        neighbor = neighbor_candidate[neighbor_ind, :, :]

        neighbor = neighbor.reshape(-1, neighbor.shape[-1]) # ((|neighbor_ind| * n) * d) array

        return nn_op.average(neighbor)