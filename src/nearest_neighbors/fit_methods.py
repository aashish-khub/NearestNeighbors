from .nnimputer import FitMethod, DataType, NearestNeighborImputer
from .estimation_methods import DREstimator, TSEstimator
import numpy.typing as npt
import numpy as np
from hyperopt import hp, fmin, tpe
from typing import cast


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
            mask_array[row, col] == 0
            or np.isnan(data_array[row, col])
            or data_array[row, col] is None
        ):
            raise ValueError("Validation cell is missing.")
        mask_array[row, col] = 0  # Set the mask to missing
    err_dists = []
    for row, col in test_cells:
        imputed_value = imputer.impute(row, col, data_array, mask_array)
        true_value = data_array[row, col]
        error = data_type.distance(imputed_value, true_value)
        err_dists.append(error)

    # Reset the mask
    for row, col in test_cells:
        mask_array[row, col] = 1

    # return error / len(test_cells)
    return float(np.nanmean(err_dists))


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
            verbose=False,
            space=hp.uniform("distance_threshold", lower_bound, upper_bound),
            algo=tpe.suggest,
            max_evals=self.n_trials,
        )
        if best_distance_threshold is None:
            return float("nan")

        return best_distance_threshold["distance_threshold"]


class DualThresholdLeaveBlockOutValidation(FitMethod):
    """An abstract base subclass for fit methods that leave out a block of cells and tune separate row and column thresholds."""

    expected_estimator_type = type(None)

    def __init__(
        self,
        block: list[tuple[int, int]],
        distance_threshold_range_row: tuple[float, float],
        distance_threshold_range_col: tuple[float, float],
        n_trials: int,
        data_type: DataType,
    ):
        """Initialize the dual threshold block fit method.

        Args:
            block (list[tuple[int, int]]): List of cells as tuples of row and column indices.
            distance_threshold_range_row (tuple[float, float]): Range of row distance thresholds to test.
            distance_threshold_range_col (tuple[float, float]): Range of column distance thresholds to test.
            n_trials (int): Number of trials to run.
            data_type (DataType): Data type to use (e.g. scalars, distributions).

        """
        self.block = block
        self.distance_threshold_range_row = distance_threshold_range_row
        self.distance_threshold_range_col = distance_threshold_range_col
        self.n_trials = n_trials
        self.data_type = data_type

    def fit(
        self,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        imputer: NearestNeighborImputer,
    ) -> tuple[float, float]:
        """Find the best distance thresholds for rows and columns by leaving out a block of cells and testing imputation.

        Args:
            data_array (npt.NDArray): Data matrix.
            mask_array (npt.NDArray): Mask matrix.
            imputer (NearestNeighborImputer): Imputer object.

        Returns:
            tuple[float, float]: Best distance thresholds for rows and columns.

        """

        def _objective(params: dict[str, float]) -> float:
            """Objective function for hyperopt.

            Args:
                params (dict[str, float]): Dictionary containing row and column distance thresholds

            Returns:
                float: Average imputation error

            """
            row_threshold = params["distance_threshold_row"]
            col_threshold = params["distance_threshold_col"]
            imputer.distance_threshold = (row_threshold, col_threshold)
            return evaluate_imputation(
                data_array, mask_array, imputer, self.block, self.data_type
            )

        lower_bound_row, upper_bound_row = self.distance_threshold_range_row
        lower_bound_col, upper_bound_col = self.distance_threshold_range_col
        best_params = fmin(
            fn=_objective,
            space={
                "distance_threshold_row": hp.uniform(
                    "distance_threshold_row", lower_bound_row, upper_bound_row
                ),
                "distance_threshold_col": hp.uniform(
                    "distance_threshold_col", lower_bound_col, upper_bound_col
                ),
            },
            algo=tpe.suggest,
            max_evals=self.n_trials,
            verbose=False,
        )

        if best_params is None:
            return float("nan"), float("nan")

        imputer.distance_threshold = (
            best_params["distance_threshold_row"],
            best_params["distance_threshold_col"],
        )

        return imputer.distance_threshold


class DRLeaveBlockOutValidation(DualThresholdLeaveBlockOutValidation):
    """Fit method by leaving out a block of cells using separate thresholds for rows and columns with a DREstimator."""

    expected_estimator_type = DREstimator

    def fit(
        self,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        imputer: NearestNeighborImputer,
    ) -> tuple[float, float]:
        """Find the best distance thresholds for rows and columns using a DREstimator.

        Args:
            data_array (npt.NDArray): Data matrix.
            mask_array (npt.NDArray): Mask matrix.
            imputer (NearestNeighborImputer): Imputer object.

        Returns:
            tuple[float, float]: Best distance thresholds for rows and columns.

        Raises:
            ValueError: If the imputer does not use a DREstimator.

        """
        if not isinstance(imputer.estimation_method, DREstimator):
            raise ValueError(
                f"The imputer must use a DREstimator for {self.__class__.__name__}."
            )
        imputer.estimation_method = cast(DREstimator, imputer.estimation_method)
        return super().fit(data_array, mask_array, imputer)


class TSLeaveBlockOutValidation(DualThresholdLeaveBlockOutValidation):
    """Fit method by leaving out a block of cells using separate thresholds for rows and columns with a TSEstimator."""

    expected_estimator_type = TSEstimator

    def fit(
        self,
        data_array: npt.NDArray,
        mask_array: npt.NDArray,
        imputer: NearestNeighborImputer,
    ) -> tuple[float, float]:
        """Find the best distance thresholds for rows and columns using a TSEstimator.

        Args:
            data_array (npt.NDArray): Data matrix.
            mask_array (npt.NDArray): Mask matrix.
            imputer (NearestNeighborImputer): Imputer object.

        Returns:
            tuple[float, float]: Best distance thresholds for rows and columns.

        Raises:
            ValueError: If the imputer does not use a TSEstimator.

        """
        if not isinstance(imputer.estimation_method, TSEstimator):
            raise ValueError(
                f"The imputer must use a TSEstimator for {self.__class__.__name__}."
            )
        imputer.estimation_method = cast(TSEstimator, imputer.estimation_method)
        return super().fit(data_array, mask_array, imputer)
