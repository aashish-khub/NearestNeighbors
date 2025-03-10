from .nnimputer import DataType
import numpy.typing as npt
from typing import Any, Callable
import numpy as np


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
        return np.mean(object_list)


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
