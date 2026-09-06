"""N^2: nearest neighbor methods for scalar and distributional matrix completion.

The public API is re-exported here so that the documented entry points can be
imported directly from the top-level package, e.g.::

    from nsquared import row_row, ts_nn, Scalar, LeaveBlockOutValidation

See https://github.com/aashish-khub/NearestNeighbors/blob/main/docs/index.md for
the full documentation.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("nsquared")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

# Core abstractions and the composite imputer.
from .nnimputer import DataType, EstimationMethod, FitMethod, NearestNeighborImputer

# Entry geometries (Distance / Average modules).
from .data_types import (
    Scalar,
    DistributionKernelMMD,
    DistributionWassersteinSamples,
    DistributionWassersteinQuantile,
)

# Estimators.
from .estimation_methods import (
    RowRowEstimator,
    ColColEstimator,
    TSEstimator,
    DREstimator,
    AWNNEstimator,
    AutoEstimator,
)

from .nadaraya_watson import NadarayaWatsonEstimator

# Convenience constructors for the common scalar imputers.
from .vanilla_nn import row_row, col_col
from .ts_nn import ts_nn
from .dr_nn import dr_nn
from .aw_nn import aw_nn

# Cross-validation / hyperparameter fitting.
from .fit_methods import (
    evaluate_imputation,
    LeaveBlockOutValidation,
    DualThresholdLeaveBlockOutValidation,
    DRLeaveBlockOutValidation,
    TSLeaveBlockOutValidation,
    AutoDRTSLeaveBlockOutValidation,
)

# Benchmark data loaders.
from .datasets.dataloader_factory import (
    NNData,
    get_available_datasets,
    register_dataset,
)
from .datasets.dataloader_base import NNDataLoader

from . import utils  # noqa: F401
from . import simulations  # noqa: F401

__all__ = [
    "__version__",
    # Core abstractions
    "DataType",
    "EstimationMethod",
    "FitMethod",
    "NearestNeighborImputer",
    # Data types
    "Scalar",
    "DistributionKernelMMD",
    "DistributionWassersteinSamples",
    "DistributionWassersteinQuantile",
    # Estimation methods
    "RowRowEstimator",
    "ColColEstimator",
    "TSEstimator",
    "DREstimator",
    "AWNNEstimator",
    "AutoEstimator",
    "NadarayaWatsonEstimator",
    # Constructors
    "row_row",
    "col_col",
    "ts_nn",
    "dr_nn",
    "aw_nn",
    # Fit methods
    "evaluate_imputation",
    "LeaveBlockOutValidation",
    "DualThresholdLeaveBlockOutValidation",
    "DRLeaveBlockOutValidation",
    "TSLeaveBlockOutValidation",
    "AutoDRTSLeaveBlockOutValidation",
    # Datasets
    "NNData",
    "NNDataLoader",
    "get_available_datasets",
    "register_dataset",
    # Subpackages
    "utils",
    "simulations",
]
