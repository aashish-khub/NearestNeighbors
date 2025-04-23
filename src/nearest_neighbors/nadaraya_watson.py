"""Implementation of Nadaraya-Watson estimation method.

Instead of identifying the nearest neighbors using a thresholding rule
(i.e., neighbors within a certain distance), the NW estimator uses all
neighbors, but weights their contribution according to a kernel function.

TODO (Albert): Test NW on HeartSteps dataset with various kernel functions
TODO (Albert): Implement additional kernel functions (e.g., Epanechnikov)
"""

from .utils.kernels import gaussian, laplace, singular_box, box
from .nnimputer import EstimationMethod, DataType
from typing import Union, Tuple

import numpy.typing as npt
import numpy as np


class NadarayaWatsonEstimator(EstimationMethod):
    valid_kernels = ["gaussian", "laplace", "singular_box", "box"]

    def __init__(self, kernel: str = "gaussian"):
        """Initialize the Nadaraya-Watson estimator.

        Args:
            kernel (str, optional): The kernel to use. Defaults to "gaussian".

        Raises:
            ValueError: If the kernel is not valid.

        """
        self.kernel = kernel
        if self.kernel not in self.valid_kernels:
            raise ValueError(
                f"{self.kernel=} is not a valid kernel. Currently supported kernels are {', '.join(self.valid_kernels)}"
            )

        super().__init__()

    def __str__(self):
        return f"NadarayaWatsonEstimator(kernel={self.kernel})"

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

        Parameters
        ----------
        row : int
            The row index of the missing value.
        column : int
            The column index of the missing value.
        data_array : npt.NDArray
            The data array.
        mask_array : npt.NDArray
            The mask array.
        distance_threshold : float
            The distance threshold.
        data_type : DataType
            The data type.

        Returns
        -------
        npt.NDArray
            The imputed value.

        """
        if isinstance(distance_threshold, tuple):
            raise ValueError(
                "The Nadaraya-Watson estimator only accepts a single distance threshold."
            )

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

        match self.kernel:
            case "gaussian":
                K = gaussian(dists=row_distances, eta=distance_threshold)
            case "laplace":
                K = laplace(dists=row_distances, eta=distance_threshold)
            case "singular_box":
                K = singular_box(dists=row_distances, eta=distance_threshold)
            case "box":
                K = box(dists=row_distances, eta=distance_threshold)
            case _:
                raise ValueError(f"{self.kernel=} is not supported")

        assert K.shape == (n_rows,)
        K = K.reshape(-1, 1)

        K_sum = K.sum(axis=0)
        K_sum_nan = np.where(K_sum == 0, np.nan, K_sum)
        # Apply mask_array to data_array
        masked_data_array = np.where(mask_array, data_array, np.nan)
        # set y to be the column-th column of masked_data_array
        y = masked_data_array[:, column]
        assert y.shape == (n_rows,)
        pred = y @ K / K_sum_nan

        # get index of nan in K along axis=0
        nan_idx = np.argwhere(np.isnan(K))
        for i, j in nan_idx:
            # NOTE: for singular kernels, anytime the row-th object is one of the
            # objects in the training set (indicated by nan values of K),
            # we set the corresponding entry of pred to the actual value of y
            #  at the row-th object
            pred[j] = y[i]
        # if an entry of K_sum is zero, set the corresponding entry of pred to zero
        pred = np.where(K_sum == 0, 0, pred)

        # NOTE: pred is a scalar, so we need to return a scalar
        (pred,) = pred
        return pred
