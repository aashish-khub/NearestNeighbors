"""Implementation of Nadaraya-Watson estimation method.

Instead of identifying the nearest neighbors using a thresholding rule
(i.e., neighbors within a certain distance), the NW estimator uses all
neighbors, but weights their contribution according to a kernel function.

TODO (Albert): Test NW on HeartSteps dataset with various kernel functions
TODO (Albert): Implement additional kernel functions (e.g., Epanechnikov)
"""

from .utils.kernels import gaussian, laplace, singular_box, box
from .nnimputer import EstimationMethod, DataType
from .estimation_methods import RowRowEstimator
from .data_types import Scalar
from typing import Union, Tuple, Any, cast

import numpy.typing as npt
import numpy as np
import warnings

__all__ = ["NadarayaWatsonEstimator"]


def _scalar_estimate(value: Any) -> npt.NDArray:
    """Wrap a scalar estimate in the return type shared by the estimators.

    RowRowEstimator returns whatever ``data_type.average`` produces, which for
    Scalar is an np.float64. We match that at runtime and cast so the signature
    inherited from EstimationMethod.impute still holds.

    Args:
        value (Any): The scalar estimate

    Returns:
        npt.NDArray: The estimate as an np.float64

    """
    return cast(npt.NDArray, np.float64(value))


class NadarayaWatsonEstimator(EstimationMethod):
    """Estimate the missing value using a kernel-weighted average over all rows.

    The row-row geometry is the same as :class:`RowRowEstimator`: the distance
    between two rows is the mean of the per-column distances over the columns
    observed in both. Rather than averaging the rows that fall within a
    threshold, this estimator averages *every* row whose target entry is
    observed, weighting each one by a kernel applied to that distance.
    """

    valid_kernels = ["gaussian", "laplace", "singular_box", "box"]

    def __init__(self, kernel: str = "gaussian", is_percentile: bool = True):
        """Initialize the Nadaraya-Watson estimator.

        Args:
            kernel (str, optional): The kernel to use. Defaults to "gaussian".
            is_percentile (bool, optional): Whether to interpret the distance
                threshold as a quantile of the observed distances rather than a
                raw kernel bandwidth. Defaults to True.

        Raises:
            ValueError: If the kernel is not valid.

        """
        if kernel not in self.valid_kernels:
            raise ValueError(
                f"{kernel=} is not a valid kernel. Currently supported kernels are {', '.join(self.valid_kernels)}"
            )
        super().__init__(is_percentile)
        self.kernel = kernel
        # NOTE: the distances are exactly the row-row distances, so we delegate to
        # that implementation (and inherit its vectorized Scalar fast path) instead
        # of duplicating it. Mirrors how ColColEstimator wraps RowRowEstimator.
        self._row_row = RowRowEstimator(is_percentile=is_percentile)
        # Alias the same dict object, so entries cached by the delegate are visible here.
        self.row_distances = self._row_row.row_distances

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
        allow_self_neighbor: bool = False,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Impute the missing value at the given row and column.

        Args:
            row (int): Row index
            column (int): Column index
            data_array (npt.NDArray): Data matrix
            mask_array (npt.NDArray): Mask matrix
            distance_threshold (float): Kernel bandwidth, or the quantile of the
                observed distances to use as the bandwidth if ``is_percentile``.
            data_type (DataType): Data type to use. Must be Scalar.
            allow_self_neighbor (bool): Whether to allow self-neighbor. Defaults to False.
            **kwargs (Any): Additional keyword arguments

        Returns:
            npt.NDArray: Imputed value, or np.nan if no row can contribute.

        Raises:
            ValueError: If given a pair of thresholds, or a non-scalar data type.

        """
        if isinstance(distance_threshold, tuple):
            raise ValueError(
                "The Nadaraya-Watson estimator only accepts a single distance threshold."
            )
        if not isinstance(data_type, Scalar):
            raise ValueError(
                "The Nadaraya-Watson estimator forms a kernel-weighted average of the "
                "target column, which is only defined for scalar outcomes. Got "
                f"{type(data_type).__name__}."
            )

        n_rows = data_array.shape[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self._calculate_distances(row, column, data_array, mask_array, data_type)
            all_dists = np.copy(self.row_distances[row])
            # Exclude the target column, so the entry being imputed does not leak
            # into the distance used to weight it.
            if not allow_self_neighbor:
                all_dists[:, column] = np.nan
            # Mean distance over the columns observed in both rows; np.nan for rows
            # that share no column with the target row.
            row_dists = np.nanmean(all_dists, axis=1)

        # Only rows whose target entry is actually observed can contribute a value.
        candidates = np.asarray(mask_array[:, column]).astype(bool) & np.isfinite(
            row_dists
        )
        if not np.any(candidates):
            return _scalar_estimate(np.nan)

        y = np.asarray(data_array[:, column], dtype=np.float64)
        eta = self._resolve_bandwidth(row_dists[candidates], distance_threshold)

        # As eta -> 0+ every supported kernel concentrates all of its mass on the
        # distance-zero rows, so take that limit directly rather than calling the
        # kernels with a non-positive bandwidth (which they reject).
        if not np.isfinite(eta) or eta <= 0:
            exact = candidates & (row_dists == 0)
            if not np.any(exact):
                return _scalar_estimate(np.nan)
            return _scalar_estimate(np.mean(y[exact]))

        weights = np.zeros(n_rows, dtype=np.float64)
        # NOTE: box and singular_box scale `dists` in place, so hand them a copy.
        weights[candidates] = self._apply_kernel(np.copy(row_dists[candidates]), eta)

        # The singular kernels return np.nan at distance zero to flag an exact match.
        # When the target row coincides with a training row, the estimate is that
        # row's observed outcome.
        exact = candidates & np.isnan(weights)
        if np.any(exact):
            return _scalar_estimate(np.mean(y[exact]))

        total = weights.sum()
        if total == 0:
            # No candidate row falls inside the bandwidth.
            return _scalar_estimate(np.nan)
        return _scalar_estimate(weights @ y / total)

    def _resolve_bandwidth(
        self,
        candidate_dists: npt.NDArray,
        distance_threshold: float,
    ) -> float:
        """Resolve the kernel bandwidth from the distance threshold.

        Args:
            candidate_dists (npt.NDArray): Finite distances of the contributing rows
            distance_threshold (float): Raw bandwidth, or a quantile in [0, 1]
                if ``is_percentile``

        Returns:
            float: The kernel bandwidth

        """
        if not self.is_percentile:
            return float(distance_threshold)
        if candidate_dists.size == 0:
            return float("nan")
        return float(np.quantile(candidate_dists, distance_threshold))

    def _apply_kernel(self, dists: npt.NDArray, eta: float) -> npt.NDArray:
        """Apply the configured kernel to a vector of distances.

        Args:
            dists (npt.NDArray): Distances to weight
            eta (float): Kernel bandwidth

        Returns:
            npt.NDArray: Kernel weights

        Raises:
            ValueError: If the configured kernel is not supported.

        """
        match self.kernel:
            case "gaussian":
                return gaussian(dists=dists, eta=eta)
            case "laplace":
                return laplace(dists=dists, eta=eta)
            case "singular_box":
                return singular_box(dists=dists, eta=eta)
            case "box":
                return box(dists=dists, eta=eta)
            case _:
                raise ValueError(f"{self.kernel=} is not supported")

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
        self._row_row._calculate_distances(row, col, data_array, mask_array, data_type)
