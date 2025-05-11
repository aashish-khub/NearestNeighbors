from .nnimputer import DataType
import numpy.typing as npt
from typing import Any, Callable
import numpy as np
import warnings


class Scalar(DataType):
    """Data type for scalars."""

    def distance(self, obj1: float, obj2: float) -> float:
        """Calculate the distance between two scalars.

        Args:
            obj1 (float): Scalar 1
            obj2 (float): Scalar 2

        Returns:
            float: Distance between the two scalars

        """
        return (obj1 - obj2) ** 2

    def average(self, object_list: npt.NDArray[Any]) -> Any:
        """Calculate the average of a list of scalars.

        Args:
            object_list (list[Any]): List of scalars

        Returns:
            Any: Average of the scalars

        """
        # Compute the mean using only the non-nan values.
        # If all values are nan, return nan.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(object_list)


class DistributionKernelMMD(DataType):
    """Data type for distributions using Kernel MMD."""

    def __init__(self, kernel: str):
        """Initialize the distribution data type with a kernel.

        Args:
            kernel (str): Kernel to use for the MMD

        """
        supported_kernels = ["linear", "square", "exponential"]

        if kernel not in supported_kernels:
            raise ValueError(
                f"Kernel {kernel} is not supported. Supported kernels are {supported_kernels}"
            )

        self.kernel = kernel

    def distance(self, obj1: npt.NDArray, obj2: npt.NDArray) -> float:
        """Calculate the distance between two distributions using Kernel MMD.

        Args:
            obj1 (npt.NDArray): (n, d) array of n samples of dimension d
            obj2 (npt.NDArray): (m, d) array of m samples of dimension d

        Returns:
            float: U-statistic of the squared MMD distance between the two distributions

        """
        m = obj1.shape[0]
        n = obj2.shape[0]

        assert obj1.shape[1] == obj2.shape[1]

        XX = np.matmul(obj1, np.transpose(obj1))  # m by m matrix with x_i^Tx_j
        YY = np.matmul(obj2, np.transpose(obj2))  # n by n matrix with y_i^Ty_j
        XY = np.matmul(obj1, np.transpose(obj2))  # m by n matrix with x_i^Ty_j

        if self.kernel == "linear":
            kXX, kYY, kXY = XX, YY, XY
        elif self.kernel == "square":
            kXX, kYY, kXY = (
                (XX + np.ones((m, m))) ** 2,
                (YY + np.ones((n, n))) ** 2,
                (XY + np.ones((m, n))) ** 2,
            )
        elif self.kernel == "exponential":
            dXX_mm = np.vstack(
                (np.diag(XX),) * m
            )  # m*m matrix : each row is the diagonal x_i^Tx_i
            dYY_nn = np.vstack(
                (np.diag(YY),) * n
            )  # n*n matrix : each row is the diagonal y_i^Ty_i
            dXX_mn = np.vstack(
                (np.diag(XX),) * n
            ).transpose()  # m*n matrix : each row is the diagonal x_i^Tx_i
            dYY_mn = np.vstack(
                (np.diag(YY),) * m
            )  # m*n matrix : each row is the diagonal y_i^Ty_i

            kXX = np.exp(-0.5 * (dXX_mm + dXX_mm.transpose() - 2 * XX))
            kYY = np.exp(-0.5 * (dYY_nn + dYY_nn.transpose() - 2 * YY))
            kXY = np.exp(-0.5 * (dXX_mn + dYY_mn - 2 * XY))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")

        val = (
            (np.nansum(kXX) - np.nansum(np.diag(kXX))) / (m * (m - 1))
            + (np.nansum(kYY) - np.nansum(np.diag(kYY))) / (n * (n - 1))
            - 2 * np.nansum(kXY) / (n * m)
        )
        if val < 0:
            val = 0

        return val

    def average(self, object_list: npt.NDArray) -> npt.NDArray:
        """Calculate the average of a list of distributions.

        Args:
            object_list (npt.NDArray[npt.NDArray]): List of distributions

        Returns:
            npt.NDArray: Average of the distributions
            (Returns is a mixture of vectors regardless of the kernel)

        """
        # Convert to list of arrays before using vstack
        arrays_to_stack = [arr for arr in object_list.flatten()]
        mixture = np.vstack(arrays_to_stack)
        return mixture


class DistributionWassersteinSamples(DataType):
    """Data type for distributions using Wasserstein distance
    where distributions are made with samples with the same number of samples.
    """

    def distance(self, obj1: npt.NDArray, obj2: npt.NDArray) -> float:
        """Calculate the Wasserstein distance between two distributions
        with the same number of samples.

        obj1 and obj2 should be 1-dimensional numpy arrays that represent
        empirical distributions.

        Args:
            obj1 (npt.NDArray): Distribution 1
            obj2 (npt.NDArray): Distribution 2

        Returns:
            float: Wasserstein distance between the two distributions

        """
        if len(obj1) != len(obj2):
            raise ValueError("Distributions must have the same number of samples")

        return float(np.mean(np.sum((np.sort(obj1) - np.sort(obj2)) ** 2)))

    def average(self, object_list: npt.NDArray[Any]) -> npt.NDArray:
        """Calculate the average of a list of distributions with the same
        number of samples.

        Args:
            object_list (npt.NDArray[Any]): List of distributions

        Returns:
            np.ndarray: Average of the distributions

        """
        return np.mean(np.sort(object_list, axis=1), axis=0)


class DistributionWassersteinQuantile(DataType):
    """Data type for distributions using Wasserstein distance
    where distributions are given by their quantile functions.
    """

    def empirical_quantile_function(
        self, samples: npt.NDArray
    ) -> Callable[[npt.NDArray], npt.NDArray]:
        """Create the quantile function of a distribution given samples.

        Args:
            samples (npt.NDArray): Samples of the distribution

        Returns:
            Callable[[npt.NDArray], npt.NDArray]: Quantile function of the distribution

        """
        samples_diff = np.concatenate(
            [np.array(samples[0]).reshape(1), np.diff(samples)]
        )

        def quantile_function(q: npt.NDArray) -> npt.NDArray:
            """Quantile function of the distribution.

            Args:
                q (npt.NDArray): Values between 0 and 1

            Returns:
                npt.NDArray: Quantile values

            """
            # Compute the empirical CDF values
            n = len(samples)
            cdf = np.arange(1, n + 1) / n
            # Use broadcasting to calculate the Heaviside contributions
            heaviside_matrix = np.heaviside(
                np.expand_dims(q, 1) - np.expand_dims(cdf, 0), 0.0
            )
            # Add a column of ones to the left of the Heaviside matrix
            first_col = np.ones(heaviside_matrix.shape[0]).reshape(-1, 1)
            heaviside_matrix = np.concatenate([first_col, heaviside_matrix], axis=1)
            # Remove the last column of Heaviside_matrix
            heaviside_matrix = heaviside_matrix[:, :-1]
            # Compute quantile values by summing contributions
            quantile_values = heaviside_matrix @ samples_diff

            return quantile_values

        return quantile_function

    def distance(
        self,
        obj1: Callable[[npt.NDArray], npt.NDArray],
        obj2: Callable[[npt.NDArray], npt.NDArray],
    ) -> float:
        """Calculate the Wasserstein distance between two distributions
        with the same number of samples.

        Args:
            obj1 (Callable[[npt.NDArray], npt.NDArray]): Distribution 1's quantile function
            obj2 (Callable[[npt.NDArray], npt.NDArray]): Distribution 2's quantile function

        Returns:
            float: Wasserstein distance between the two distributions

        """
        x = np.linspace(0, 1, 1000)
        return float(np.trapezoid((obj1(x) - obj2(x)) ** 2, x=x))

    def average(self, object_list: npt.NDArray) -> Callable:
        """Calculate the average of a list of quantile functions

        Args:
            object_list (npt.NDArray[Any]): List of quantile functions

        Returns:
            Callable: Average quantile function

        """
        # filter out float values
        object_list = np.array(
            [fn for fn in object_list if not isinstance(fn, float)], dtype=object
        )
        if len(object_list) == 0:
            return lambda x: x * 0

        def lin_comb_fn(quantiles: npt.NDArray) -> npt.NDArray:
            """Average a bunch of quantile functions.

            Args:
                quantiles (npt.NDArray): Values between 0 and 1

            Returns:
                npt.NDArray: Quantile values

            """
            # Compute the quantile values for each function
            quantile_values = np.stack([fn(quantiles) for fn in object_list])
            lin_comb_values = np.sum(quantile_values, axis=0) / len(object_list)
            return lin_comb_values

        return lin_comb_fn
