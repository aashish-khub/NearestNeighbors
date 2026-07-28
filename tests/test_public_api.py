"""Tests that the documented public API stays importable.

``docs/api_reference.md`` promises that every documented name is importable from
the top-level ``nsquared`` package. These tests pin that promise so a refactor
of ``nsquared/__init__.py`` cannot silently break the documentation.
"""

import numpy as np
import pytest

import nsquared

DOCUMENTED_NAMES = [
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
]


@pytest.mark.parametrize("name", DOCUMENTED_NAMES)
def test_documented_name_is_importable(name: str) -> None:
    """Each documented name is exported from the top-level package."""
    assert hasattr(nsquared, name), f"nsquared.{name} is documented but not exported"
    assert name in nsquared.__all__, f"nsquared.{name} is missing from __all__"


def test_all_is_consistent_with_the_module() -> None:
    """Nothing is listed in __all__ that does not actually exist."""
    for name in nsquared.__all__:
        assert hasattr(nsquared, name), f"__all__ lists {name}, which does not exist"


def test_constructors_return_usable_imputers() -> None:
    """The convenience constructors build imputers that actually impute."""
    data = np.full((6, 6), 2.0)
    mask = np.ones((6, 6), dtype=int)

    for imputer in [
        nsquared.row_row(distance_threshold=1.0, is_percentile=False),
        nsquared.col_col(distance_threshold=1.0, is_percentile=False),
        nsquared.ts_nn(distance_threshold_row=1.0, distance_threshold_col=1.0),
        nsquared.dr_nn(distance_threshold_row=1.0, distance_threshold_col=1.0),
    ]:
        value = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
        assert value == pytest.approx(2.0), f"{imputer} failed on a constant matrix"


def test_custom_data_type_works_with_every_estimator() -> None:
    """A user-defined DataType composes with the shipped estimators.

    This is the extension guarantee documented in ``docs/user_guide.md``: define
    a distance and an average, and every geometry-agnostic estimator works.
    """

    class L1Scalar(nsquared.DataType):
        """Scalars under absolute rather than squared error."""

        def distance(self, obj1: float, obj2: float) -> float:
            """Absolute difference between two scalars."""
            return float(abs(obj1 - obj2))

        def average(self, object_list: np.ndarray) -> float:
            """Median, the Frechet mean under absolute error."""
            return float(np.nanmedian(object_list))

    data = np.full((6, 6), 5.0)
    mask = np.ones((6, 6), dtype=int)

    for estimator, threshold in [
        (nsquared.RowRowEstimator(is_percentile=False), 1.0),
        (nsquared.ColColEstimator(is_percentile=False), 1.0),
        (nsquared.TSEstimator(is_percentile=False), (1.0, 1.0)),
    ]:
        imputer = nsquared.NearestNeighborImputer(estimator, L1Scalar(), threshold)
        value = imputer.impute(row=0, column=0, data_array=data, mask_array=mask)
        assert value == pytest.approx(5.0)


if __name__ == "__main__":
    pytest.main()
