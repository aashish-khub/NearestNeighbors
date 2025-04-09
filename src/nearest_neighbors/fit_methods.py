from .nnimputer import FitMethod, DataType, NearestNeighborImputer
import numpy.typing as npt
import numpy as np
from hyperopt import hp, fmin, tpe


def evaluate_imputation(
    data_array: npt.NDArray,
    mask_array: npt.NDArray,
    imputer: NearestNeighborImputer,
    test_cells: list[tuple[int, int]],
    data_type: DataType,
) -> float:
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
    # Block out the test cells
    for row, col in test_cells:
        if (
            (mask_array[row, col] == 0)
            | (np.all(np.isnan(data_array[row, col])))  # TODO: review for distributions
            | (data_array[row, col] is None)
        ):
            raise ValueError("Validation cell is missing.")
        mask_array[row, col] = 0  # Set the mask to missing

    for row, col in test_cells:
        imputed_value = imputer.impute(row, col, data_array, mask_array)
        true_value = data_array[row, col]
        # print(imputed_value)
        # print(true_value)
        # print(f"Imputed value: {imputed_value}, True value: {true_value}")
        error += data_type.distance(imputed_value, true_value)
        # print(f"Error: {error}")
        # print(error)

    # Reset the mask
    for row, col in test_cells:
        mask_array[row, col] = 1

    return error / len(test_cells)


class LeaveBlockOutValidation(FitMethod):
    """Fit method by leaving out a block of cells."""

    def __init__(
        self,
        block: list[tuple[int, int]],
        distance_threshold_range: tuple[float, float],
        n_trials: int,
        data_type: DataType,
    ):
        """Initialize the block fit method.

        Args:
            block (list[tuple[int,int]]): List of cells as tuples of row and column indices
            distance_threshold_range (tuple[float,float]): Range of distance thresholds to test
            n_trials (int): Number of trials to run
            data_type (DataType): Data type to use (e.g. scalars, distributions)

        """
        self.block = block
        self.distance_threshold_range = distance_threshold_range
        self.n_trials = n_trials
        self.data_type = data_type

    def fit(
        self,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        imputer: NearestNeighborImputer,
    ) -> float:
        """Find the best distance threshold for the given data
        by leaving out a block of cells and testing imputation against them.

        Args:
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            imputer (NearestNeighborImputer): Imputer object

        Returns:
            float: Best distance threshold

        """

        def objective(distance_threshold: float) -> float:
            """Objective function for hyperopt.

            Args:
                distance_threshold (float): Distance threshold to test

            Returns:
                float: Average imputation error

            """
            imputer.distance_threshold = distance_threshold
            return evaluate_imputation(
                data_array, mask_array, imputer, self.block, self.data_type
            )

        lower_bound, upper_bound = self.distance_threshold_range
        best_distance_threshold = fmin(
            fn=objective,
            verbose=True,
            space=hp.uniform("distance_threshold", lower_bound, upper_bound),
            algo=tpe.suggest,
            max_evals=self.n_trials,
        )
        if best_distance_threshold is None:
            return float("nan")
        imputer.distance_threshold = best_distance_threshold["distance_threshold"]
        return best_distance_threshold["distance_threshold"]
