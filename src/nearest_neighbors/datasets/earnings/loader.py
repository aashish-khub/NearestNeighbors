import numpy as np
from typing import Callable
import numpy.typing as npt
import pandas as pd
from nearest_neighbors.datasets.dataloader_base import NNDataLoader


def empirical_quantile_function(samples: npt.NDArray) -> Callable:
    """Create an empirical quantile function from the samples.

    Args:
        samples (npt.NDArray): Input samples for quantile function.

    Returns:
        Callable: Quantile function that takes a single argument and returns quantile values.

    """
    samples_diff = np.concatenate([np.array(samples[0]).reshape(1), np.diff(samples)])

    def quantile_function(q: npt.NDArray) -> npt.NDArray:
        """Compute quantile values based on the input quantile.

        Args:
            q (npt.NDArray): Quantile values to compute.

        Returns:
            npt.NDArray: Computed quantile values.

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


class EarningsDataLoader(NNDataLoader):
    """Data loader for earnings dataset."""

    def __init__(
        self,
        quarterly_data: dict,
        quarterly_actual: dict,
        ibes_tickers: list[str],
        current_time: pd.Timestamp,
        agg: str = "mean",
        save_processed: bool = False,
    ):
        """Initializes the EarningsDataLoader.

        Args:
            quarterly_data (dict): Quarterly earnings data.
            quarterly_actual (dict): Actual earnings data.
            ibes_tickers (list[str]): List of IBES tickers.
            current_time (pd.Timestamp): Current time for filtering data.
            agg (str, optional): Aggregation method. Defaults to "mean".
            save_processed (bool, optional): Whether to save processed data. Defaults to False.

        """
        super().__init__(agg=agg, save_processed=save_processed)
        self.quarterly_actual = quarterly_actual
        self.ibes_tickers = ibes_tickers
        self.current_time = current_time
        self.rows = [oftic for oftic, _ in self.ibes_tickers]
        self.cols = [
            (year, quarter) for year in range(2010, 2025) for quarter in range(1, 5)
        ]

        self.supported_aggs = ["mean", "sum", "median", "std", "variance"]
        if self.agg == "mean":
            self.agg_func = np.mean
        elif self.agg == "sum":
            self.agg_func = np.sum
        elif self.agg == "median":
            self.agg_func = np.median
        elif self.agg == "std":
            self.agg_func = np.std
        elif self.agg == "variance":
            self.agg_func = np.var
        else:
            raise ValueError(
                f"Aggregation method {self.agg} not supported. Supported methods: {self.supported_aggs}"
            )

        # Filter the data to the current time
        self.quarterly_data = dict()
        for k, v in quarterly_data.items():
            if v is None:
                self.quarterly_data[k] = None
                continue
            self.quarterly_data[k] = v[v["ann_datetime"] <= self.current_time]
            if self.quarterly_data[k].shape[0] == 0:
                self.quarterly_data[k] = None

    def process_data_scalar(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Processes the data into scalar matrix setting.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: Tuple of 2D data matrix and mask.

        """
        data_matrix = np.zeros((len(self.rows), len(self.cols)))
        mask_matrix = np.ones((len(self.rows), len(self.cols)))

        for i, oftic in enumerate(self.rows):
            for j, (year, quarter) in enumerate(self.cols):
                data = self.quarterly_data.get((oftic, year, quarter))
                if data is not None:
                    data_matrix[i, j] = self.agg_func(data["value"])
                    mask_matrix[i, j] = 1
                else:
                    data_matrix[i, j] = np.nan
                    mask_matrix[i, j] = 0

        return data_matrix, mask_matrix

    def process_data_distribution(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Processes the data into distributional setting.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: Tuple of 2D data matrix and mask.

        """
        data_matrix = np.empty((len(self.rows), len(self.cols)), dtype=object)
        mask_matrix = np.ones((len(self.rows), len(self.cols)))

        for i, oftic in enumerate(self.rows):
            for j, (year, quarter) in enumerate(self.cols):
                data = self.quarterly_data.get((oftic, year, quarter))
                if data is not None:
                    data_matrix[i, j] = empirical_quantile_function(
                        data["value"].to_numpy()
                    )
                    mask_matrix[i, j] = 1
                else:
                    data_matrix[i, j] = np.nan
                    mask_matrix[i, j] = 0

        return data_matrix, mask_matrix

    def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
        """Returns the full state of the data loader as a dictionary.

        Args:
            include_metadata (bool, optional): Whether to include metadata in the dictionary. Defaults to False.

        Returns:
            dict: Full state of the data loader.

        """
        state = {
            "quarterly_data": self.quarterly_data,
            "quarterly_actual": self.quarterly_actual,
            "ibes_tickers": self.ibes_tickers,
            "current_time": self.current_time,
        }
        if include_metadata:
            state["metadata"] = {"agg": self.agg, "save_processed": self.save_processed}
        return state
