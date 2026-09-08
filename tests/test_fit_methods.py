"""Tests for the cross-validation (``FitMethod``) layer.

Each fit method holds out a block of observed entries, imputes them, and picks
the distance threshold(s) that minimise the imputation error.
"""

import warnings
from typing import cast

import numpy as np
import pytest
from hyperopt import Trials

from nsquared import (
    AutoDRTSLeaveBlockOutValidation,
    AutoEstimator,
    DRLeaveBlockOutValidation,
    LeaveBlockOutValidation,
    NNData,
    NearestNeighborImputer,
    Scalar,
    TSLeaveBlockOutValidation,
    dr_nn,
    evaluate_imputation,
    row_row,
    ts_nn,
)

N_TRIALS = 10


def make_problem(
    seed: int = 0, size: int = 25, block_size: int = 20
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Build a synthetic matrix plus a held-out block of observed entries.

    Args:
        seed (int): Random seed for both the data and the block.
        size (int): Number of rows and columns.
        block_size (int): Number of observed entries to hold out.

    Returns:
        tuple: The data matrix, the mask, and the held-out block.

    """
    loader = NNData.create(
        "synthetic_data", num_rows=size, num_cols=size, seed=seed, miss_prob=0.3
    )
    data, mask = loader.process_data_scalar()

    rng = np.random.default_rng(seed)
    observed = np.argwhere(mask == 1)
    chosen = rng.choice(len(observed), block_size, replace=False)
    block = [(int(observed[i][0]), int(observed[i][1])) for i in chosen]
    return data, mask, block


def test_imputing_without_a_threshold_raises() -> None:
    """An untuned imputer refuses to guess rather than silently using a default."""
    data, mask, _ = make_problem()
    imputer = row_row()

    with pytest.raises(ValueError):
        imputer.impute(row=0, column=0, data_array=data, mask_array=mask)


def test_leave_block_out_sets_threshold_in_place() -> None:
    """Fit returns the chosen threshold and also stores it on the imputer."""
    data, mask, block = make_problem()
    imputer = row_row()

    cv = LeaveBlockOutValidation(
        block=block,
        distance_threshold_range=(0, 1),
        n_trials=N_TRIALS,
        data_type=Scalar(),
    )
    best = cv.fit(data, mask, imputer)

    # Without ret_trials the return type is a bare threshold.
    assert isinstance(best, float)
    assert 0 <= best <= 1
    assert imputer.distance_threshold == best

    # The imputer is now usable.
    assert np.isfinite(imputer.impute(0, 0, data, mask))


def test_leave_block_out_is_reproducible() -> None:
    """The same rng gives the same threshold."""
    data, mask, block = make_problem()

    thresholds = []
    for _ in range(2):
        imputer = row_row()
        cv = LeaveBlockOutValidation(
            block=block,
            distance_threshold_range=(0, 1),
            n_trials=N_TRIALS,
            data_type=Scalar(),
            rng=np.random.default_rng(7),
        )
        thresholds.append(cv.fit(data, mask, imputer))

    assert thresholds[0] == pytest.approx(thresholds[1])


def test_leave_block_out_can_return_trials() -> None:
    """ret_trials exposes the hyperopt search history."""
    data, mask, block = make_problem()
    imputer = row_row()

    cv = LeaveBlockOutValidation(
        block=block,
        distance_threshold_range=(0, 1),
        n_trials=N_TRIALS,
        data_type=Scalar(),
    )
    result = cv.fit(data, mask, imputer, ret_trials=True)

    # With ret_trials the return type widens to (threshold, Trials).
    assert isinstance(result, tuple)
    best, trials = result
    assert 0 <= best <= 1
    assert len(trials.trials) == N_TRIALS


@pytest.mark.parametrize(
    "constructor,cv_class",
    [(ts_nn, TSLeaveBlockOutValidation), (dr_nn, DRLeaveBlockOutValidation)],
)
def test_dual_threshold_fit_methods(constructor: object, cv_class: object) -> None:
    """Two-threshold estimators get a (row, column) threshold pair."""
    data, mask, block = make_problem()
    imputer = constructor()  # type: ignore[operator]

    cv = cv_class(  # type: ignore[operator]
        block=block,
        distance_threshold_range_row=(0, 1),
        distance_threshold_range_col=(0, 1),
        n_trials=N_TRIALS,
        data_type=Scalar(),
    )
    row_threshold, col_threshold = cv.fit(data, mask, imputer)

    assert 0 <= row_threshold <= 1
    assert 0 <= col_threshold <= 1
    assert imputer.distance_threshold == (row_threshold, col_threshold)


def test_auto_estimator_fit() -> None:
    """AutoEstimator additionally tunes its mixing weight."""
    data, mask, block = make_problem()
    imputer = NearestNeighborImputer(AutoEstimator(), Scalar())

    cv = AutoDRTSLeaveBlockOutValidation(
        block=block,
        distance_threshold_range_row=(0, 1),
        distance_threshold_range_col=(0, 1),
        alpha_range=(0, 1),
        n_trials=N_TRIALS,
        data_type=Scalar(),
    )
    cv.fit(data, mask, imputer)

    assert np.isfinite(imputer.impute(0, 0, data, mask))


def test_evaluate_imputation_is_zero_on_a_constant_matrix() -> None:
    """With every entry equal, held-out entries are imputed exactly."""
    imputer = row_row(distance_threshold=1.0, is_percentile=False)
    data = np.full((10, 10), 3.0)
    mask = np.ones((10, 10), dtype=int)
    block = [(0, 0), (1, 1), (2, 3)]

    error = evaluate_imputation(data, mask, imputer, block, Scalar())
    assert error == pytest.approx(0.0)


def test_tuning_beats_a_bad_threshold() -> None:
    """Cross-validation finds a threshold at least as good as an arbitrary one."""
    data, mask, block = make_problem(seed=3)

    tuned = row_row()
    LeaveBlockOutValidation(
        block=block,
        distance_threshold_range=(0, 1),
        n_trials=25,
        data_type=Scalar(),
        rng=np.random.default_rng(0),
    ).fit(data, mask, tuned)
    tuned_error = evaluate_imputation(data, mask, tuned, block, Scalar())

    untuned = row_row(distance_threshold=1.0, is_percentile=True)
    untuned_error = evaluate_imputation(data, mask, untuned, block, Scalar())

    assert tuned_error <= untuned_error


if __name__ == "__main__":
    pytest.main()


@pytest.mark.parametrize("seed", [0, 1])
def test_single_threshold_search_is_reproducible_with_rng(seed: int) -> None:
    """Two searches with equally seeded generators pick the same threshold."""
    data, mask, block = make_problem(seed)

    def run() -> float:
        cv = LeaveBlockOutValidation(
            block=block,
            distance_threshold_range=(0, 1),
            n_trials=N_TRIALS,
            data_type=Scalar(),
            rng=np.random.default_rng(seed),
        )
        result = cv.fit(data, mask, row_row())
        assert isinstance(result, float)
        return result

    assert run() == run()


def test_dual_threshold_search_is_reproducible_with_rng() -> None:
    """The two-threshold fitters accept ``rng`` and honour it."""
    data, mask, block = make_problem()

    def run_ts() -> tuple[float, float]:
        cv = TSLeaveBlockOutValidation(
            block=block,
            distance_threshold_range_row=(0, 1),
            distance_threshold_range_col=(0, 1),
            n_trials=N_TRIALS,
            data_type=Scalar(),
            rng=np.random.default_rng(3),
        )
        return cv.fit(data, mask, ts_nn())  # type: ignore[return-value]

    def run_auto() -> tuple[float, float]:
        imputer = NearestNeighborImputer(AutoEstimator(), Scalar())
        cv = AutoDRTSLeaveBlockOutValidation(
            block=block,
            distance_threshold_range_row=(0, 1),
            distance_threshold_range_col=(0, 1),
            alpha_range=(0, 1),
            n_trials=N_TRIALS,
            data_type=Scalar(),
            rng=np.random.default_rng(3),
        )
        cv.fit(data, mask, imputer)
        return imputer.distance_threshold, imputer.estimation_method.alpha  # type: ignore[return-value]

    assert run_ts() == run_ts()
    assert run_auto() == run_auto()


def test_ts_fit_returns_trials_when_asked() -> None:
    """Regression test: ``ret_trials`` was silently dropped by the TS fitter."""
    data, mask, block = make_problem()
    cv = TSLeaveBlockOutValidation(
        block=block,
        distance_threshold_range_row=(0, 1),
        distance_threshold_range_col=(0, 1),
        n_trials=N_TRIALS,
        data_type=Scalar(),
    )
    result = cv.fit(data, mask, ts_nn(), ret_trials=True)
    assert isinstance(result, tuple) and len(result) == 2
    thresholds, trials = result
    assert isinstance(thresholds, tuple) and len(thresholds) == 2
    assert isinstance(trials, Trials)
    assert len(trials.trials) == N_TRIALS


def test_evaluate_imputation_all_nan_is_nan_without_warning() -> None:
    """When nothing can be imputed the score is nan, reported silently."""
    data = np.ones((4, 4))
    mask = np.ones((4, 4), dtype=int)

    class NothingImputer:
        def impute(self, *args: object, **kwargs: object) -> np.floating:
            return np.float64(np.nan)

    imputer = cast(NearestNeighborImputer, NothingImputer())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = evaluate_imputation(data, mask, imputer, [(0, 0), (1, 1)], Scalar())
    assert np.isnan(score)
