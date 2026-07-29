"""Tests that the documented public API stays importable.

``docs/api_reference.md`` promises that every documented name is importable from
the top-level ``nsquared`` package. These tests pin that promise so a refactor
of ``nsquared/__init__.py`` cannot silently break the documentation.
"""

import importlib.util
import os

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


def test_baselines_live_under_the_nsquared_namespace() -> None:
    """The baselines must not be installed as a top-level package.

    They used to be shipped as a top-level ``baselines`` package, which squats a
    very generic name in the importing user's namespace (OpenAI's reinforcement
    learning library is also called ``baselines``). They now live at
    ``nsquared.baselines``.
    """
    from nsquared.baselines import usvt

    assert usvt.__module__.startswith("nsquared.baselines")

    # Installing this package must not make a bare `import baselines` resolve to
    # our code. Use find_spec rather than an import statement so this stays a
    # runtime check about packaging, not a static import of a module that should
    # not exist.
    spec = importlib.util.find_spec("baselines")
    if spec is None or spec.origin is None:
        return
    assert not spec.origin.endswith(f"nsquared{os.sep}baselines{os.sep}__init__.py")


def test_usvt_needs_no_optional_dependency() -> None:
    """Usvt runs on a core-only install: it uses nothing but NumPy."""
    from nsquared.baselines import usvt

    matrix = np.array([[1.0, 2.0], [np.nan, 4.0]])
    completed = usvt(matrix)

    assert completed.shape == matrix.shape
    assert not np.any(np.isnan(completed))


def test_softimpute_is_resolved_lazily() -> None:
    """Softimpute is reachable but must not be imported eagerly.

    Importing ``nsquared.baselines`` has to work without ``fancyimpute`` so that
    a core-only install can still use ``usvt``.
    """
    import nsquared.baselines as pkg

    assert "softimpute" in pkg.__all__

    try:
        resolved = pkg.softimpute
    except ImportError as exc:
        # Without the extra, the error has to name the extra to install.
        assert "nsquared[baselines]" in str(exc)
    else:
        assert callable(resolved)

    with pytest.raises(AttributeError):
        pkg.not_a_real_baseline


def test_softimpute_stays_a_function_across_accesses() -> None:
    """Repeated access must not degrade into the implementation module.

    The lazy loader caches the resolved function on the package. If the
    implementation module were also named ``softimpute``, the import system
    would rebind the attribute to that module and every access after the first
    would return a module instead of a callable.
    """
    fancyimpute = pytest.importorskip(
        "fancyimpute", reason="requires the 'baselines' extra"
    )
    assert fancyimpute is not None

    import nsquared.baselines as pkg
    from nsquared.baselines import softimpute

    assert callable(pkg.softimpute)
    assert pkg.softimpute is pkg.softimpute
    assert pkg.softimpute is softimpute


def test_softimpute_completes_a_matrix() -> None:
    """The SoftImpute baseline runs end to end on its installed dependencies.

    Guards the ``scikit-learn<1.8`` pin in the ``baselines`` extra: fancyimpute
    0.7.0 calls ``check_array(force_all_finite=...)``, removed in scikit-learn
    1.8, so an unpinned resolve raises TypeError here.
    """
    pytest.importorskip("fancyimpute", reason="requires the 'baselines' extra")
    from nsquared.baselines import softimpute

    matrix = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    completed = softimpute(matrix)

    assert completed.shape == matrix.shape
    assert not np.any(np.isnan(completed))
    # Observed entries are preserved; the missing one is filled sensibly.
    assert completed[1, 1] == pytest.approx(5.0, abs=1.0)


if __name__ == "__main__":
    pytest.main()
