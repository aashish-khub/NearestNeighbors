from .nnimputer import FitMethod
from .nnimputer import NearestNeighborImputer
from .data_types import DistributionKernelMMD
import numpy.typing as npt
import numpy as np


class DirectOptimization(FitMethod):
    """Non-cross-validation fit method. Analytically optimizes the squared MMD error."""

    def __init__(self, row: int, column: int, kernel: str, eta_cand: npt.NDArray, delta: float):
        """Initialize the fit method with additional parameters.

        Args:
            kernel (str): Kernel to use for the MMD
            eta_cand (npt.NDArray): Candidate distance thresholds
            delta (float): Significance level
            row (int): target row index
            column (int): target column index

        """
        supported_kernels = ["exponential"]

        if kernel not in supported_kernels:
            raise ValueError(
                f"Kernel {kernel} is not supported. Supported kernels are {supported_kernels}"
            )

        self.kernel = kernel
        self.eta_cand = eta_cand
        self.delta = delta
        self.row = row
        self.column = column

    def fit(self, data_array: npt.NDArray, mask_array: npt.NDArray, imputer: NearestNeighborImputer) -> float:
        """Analytically optimizes the squared MMD error.

        Args:
            data_array (npt.NDArray): Data array
            mask_array (npt.NDArray): Mask array
            imputer (NearestNeighborImputer): Imputer

        Returns:
            float: Best distance threshold

        """
        # Initialize sup_kern outside conditional
        sup_kern = 1  # Default value
        if self.kernel == "exponential":
            sup_kern = 1  # TODO: need to change for general kernels

        delta = self.delta
        eta_cand = self.eta_cand
        row = self.row
        column = self.column

        n_rows, n_cols = data_array.shape[0], data_array.shape[1]
        n = data_array[0, 0].shape[0]  # number of samples per distribution
        data_type = DistributionKernelMMD(self.kernel)

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

        perf = []

        for eta in eta_cand:
            neighborhood = np.where((row_distances < eta) * (mask_array[:, column]) == 1)[0]  # Set of neighbors: (i) within eta distance (ii) observed

            if sum(np.isin(neighborhood, row)) == 1:  # Pretending as if (row, column) entry is missing
                neighborhood = np.delete(neighborhood, np.where(neighborhood == row)[0])

            if len(neighborhood) == 0:  # Default (null) output when there is zero neighbor
                perf.append(10**5)  # Avoid selecting such eta without neighbors
            else:
                overlap = []
                for neighbor in neighborhood:
                    overlap.append(np.sum(mask_array[row, :] * mask_array[neighbor, :]))

                Bias = 8 * np.exp(1/np.exp(1)) * sup_kern * np.log(2*n_rows/delta) / (np.sqrt(2*np.log(2)*np.min(overlap)))
                Variance = 4 * sup_kern * (np.log(n) + 1.5) / (n*len(neighborhood))

                perf.append(eta + Bias + Variance)

        if not perf:  # Handle case when perf list is empty
            return float('inf')  # Return infinity as a default value when no valid threshold is found
            
        eta_star = eta_cand[np.argmin(perf)]
        return eta_star
    



class CrossValidation(FitMethod):
    """Cross-validation fit method. Uses cross-validation to find the best distance threshold."""

    def fit(self, data_array: npt.NDArray, mask_array: npt.NDArray, imputer: NearestNeighborImputer) -> float:
        """Uses cross-validation to find the best distance threshold.
        
        Args:
            data_array (npt.NDArray): Data array
            mask_array (npt.NDArray): Mask array
            imputer (NearestNeighborImputer): Imputer

        Returns:
            float: Best distance threshold
            
        """
        return 0.0  # TODO: Implement cross-validation