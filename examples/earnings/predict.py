import pickle
from typing import Callable
import numpy as np
import pandas as pd
from nsquared.datasets.earnings.loader import EarningsDataLoader
from nsquared.estimation_methods import ColColEstimator
from nsquared.nnimputer import NearestNeighborImputer
from nsquared.fit_methods import LeaveBlockOutValidation
from nsquared.data_types import DistributionWassersteinQuantile

print("Loading downloaded files...")

with open("examples/earnings/ibes_tickers.pkl", "rb") as f:
    ibes_tickers = pickle.load(f)

# ibes_data = dict()
# for oftic, ibes in tqdm(ibes_tickers):
#     data = pd.read_csv(f"data/{oftic}.csv")
#     ibes_data[oftic] = data

with open("examples/earnings/quarterly_actual.pkl", "rb") as f:
    quarterly_actual = pickle.load(f)

with open("examples/earnings/quarterly_data.pkl", "rb") as f:
    quarterly_data = pickle.load(f)


def get_rows_cols(user_user: bool = True) -> tuple[list, list]:
    """Get the rows and columns for the quarterly earnings data.

    Args:
        user_user (bool, optional): Indicates whether to return user-user or user-item data. Defaults to True.

    Returns:
        tuple[list, list]: List of rows and column id's.

    """
    cols = [(year, quarter) for year in range(2010, 2025) for quarter in range(1, 5)]
    if user_user is True:
        return ibes_tickers, cols
    else:
        return cols, ibes_tickers


def expectation(quantile_func: Callable, N: int = 1000) -> float:
    """Calculate the expected value based on the quantile function.

    Args:
        quantile_func (Callable): The quantile function to evaluate.
        N (int, optional): Number of points for trapezoidal integration. Defaults to 1000.

    Returns:
        float: The expected value calculated from the quantile function.

    """
    x = np.linspace(0, 1, N)
    return np.trapezoid(quantile_func(x), x=x)


def var(quantile_func: Callable, N: int = 1000) -> float:
    """Calculate the variance of a quantile function.

    Args:
        quantile_func (Callable): The quantile function to evaluate.
        N (int, optional): Number of points for trapezoidal integration. Defaults to 1000.

    Returns:
        float: The calculated variance of the quantile function.

    """
    x = np.linspace(0, 1, N)
    return np.trapezoid((quantile_func(x) - expectation(quantile_func)) ** 2, x=x)


def squared_diff(
    quantile_func1: Callable, quantile_func2: Callable, N: int = 1000
) -> float:
    """Calculate the squared difference between two quantile functions.

    Args:
        quantile_func1 (Callable): The first quantile function to compare.
        quantile_func2 (Callable): The second quantile function to compare.
        N (int, optional): Number of points for trapezoidal integration. Defaults to 1000.

    Returns:
        float: The calculated squared difference between the two quantile functions.

    """
    x = np.linspace(0, 1, N)
    return np.trapezoid((quantile_func1(x) - quantile_func2(x)) ** 2, x=x)


def relative_error(est: float, true: float) -> float:
    """Calculate the relative error between estimated and true values.

    Args:
        est (float): Estimated value.
        true (float): True value.

    Returns:
        float: The calculated relative error.

    """
    return np.abs((est - true) / true) if true != 0 else np.inf


current_time = pd.Timestamp("2024-01-01")
if current_time is not pd.Timestamp:
    raise ValueError("current_time must be a pandas Timestamp object")

data_loader = EarningsDataLoader(
    ibes_tickers=ibes_tickers,
    quarterly_data=quarterly_data,
    quarterly_actual=quarterly_actual,
    current_time=current_time,
    agg="mean",
    save_processed=False,
)

# Load scalar data
# print("Loading scalar data...")
# scalar_data, scalar_mask = data_loader.process_data_scalar()

# Load distributional data
print("Loading distributional data...")
distribution_data, distribution_mask = data_loader.process_data_distribution()

estimation_method = ColColEstimator()
data_type = DistributionWassersteinQuantile()
estimator = NearestNeighborImputer(
    estimation_method=estimation_method,
    data_type=data_type,
    distance_threshold=None,
)
fit_method = LeaveBlockOutValidation(
    block=[(0, 0)],
    distance_threshold_range=(0.001, 1e3),
    n_trials=20,
    data_type=data_type,
)

# Fit the model
print("Fitting the model...")
eta = fit_method.fit(distribution_data, distribution_mask, estimator)

print(f"Best distance threshold: {eta}")

# Impute the mising values
print("Imputing the missing values...")
imputed_data = estimator.impute(0, 0, distribution_data, distribution_mask)

# # Calculate the expected value
# print("Calculating the expected value...")

# expected_value = expectation(imputed_data)

# print(f"Expected value: {expected_value}")
