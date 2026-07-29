"""Tests for the SoftImpute baseline.

``nsquared.baselines.softimpute`` is a from-scratch NumPy implementation of
Mazumder et al. (2010) and the bi-scaling of Hastie et al. (2015). It replaced a
``fancyimpute`` dependency, so these tests pin the properties that the
replacement has to keep: it recovers low-rank structure, leaves observed entries
untouched, honours its parameters, and fails loudly on input it cannot handle.

Agreement with the original ``fancyimpute`` implementation was checked
separately at the time of the swap: relative difference on the imputed entries
was below 2e-4 across low-rank, noisy, sparse, high-rank and non-square cases,
with identical RMSE against ground truth to four decimal places.
"""

import numpy as np
import pytest

from nsquared.baselines import softimpute


def low_rank_problem(
    n_rows: int = 40,
    n_cols: int = 30,
    rank: int = 3,
    noise: float = 0.1,
    missing_rate: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a noisy low-rank matrix and a copy with entries removed.

    Args:
        n_rows (int): Number of rows.
        n_cols (int): Number of columns.
        rank (int): Rank of the underlying signal.
        noise (float): Standard deviation of the additive noise.
        missing_rate (float): Fraction of entries to blank out.
        seed (int): Random seed.

    Returns:
        tuple: The ground-truth matrix and the matrix with ``np.nan`` holes.

    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(n_rows, rank)) @ rng.normal(size=(rank, n_cols))
    truth = truth + rng.normal(scale=noise, size=(n_rows, n_cols))

    observed = truth.copy()
    holes = rng.random((n_rows, n_cols)) < missing_rate
    # Keep every row and column at least partially observed.
    holes[:, 0] = False
    holes[0, :] = False
    observed[holes] = np.nan
    return truth, observed


def test_fills_every_missing_entry() -> None:
    """No nan survives the imputation."""
    _, observed = low_rank_problem()
    completed = softimpute(observed)

    assert completed.shape == observed.shape
    assert not np.any(np.isnan(completed))


def test_observed_entries_are_returned_unchanged() -> None:
    """SoftImpute estimates the holes; it must not perturb the data."""
    _, observed = low_rank_problem()
    mask = ~np.isnan(observed)

    completed = softimpute(observed)
    np.testing.assert_allclose(completed[mask], observed[mask])


def test_recovers_low_rank_structure() -> None:
    """On a genuinely low-rank matrix the estimate beats imputing the mean."""
    truth, observed = low_rank_problem(rank=3, noise=0.1)
    holes = np.isnan(observed)

    completed = softimpute(observed)
    error = np.sqrt(np.mean((completed[holes] - truth[holes]) ** 2))
    mean_baseline = np.sqrt(np.mean((np.nanmean(observed) - truth[holes]) ** 2))

    assert error < mean_baseline / 2


def test_accuracy_degrades_gracefully_with_missingness() -> None:
    """More missing data is harder, but the estimate stays informative."""
    errors = []
    for missing_rate in (0.2, 0.5, 0.8):
        truth, observed = low_rank_problem(rank=3, missing_rate=missing_rate, seed=1)
        holes = np.isnan(observed)
        completed = softimpute(observed)
        errors.append(np.sqrt(np.mean((completed[holes] - truth[holes]) ** 2)))

    assert errors[0] <= errors[1] <= errors[2]
    assert np.all(np.isfinite(errors))


def test_exact_low_rank_matrix_is_recovered_closely() -> None:
    """With no noise and a rank-1 signal, recovery should be sharp."""
    rng = np.random.default_rng(3)
    truth = np.outer(rng.normal(size=25), rng.normal(size=20))
    observed = truth.copy()
    holes = rng.random(truth.shape) < 0.25
    holes[0, :] = False
    holes[:, 0] = False
    observed[holes] = np.nan

    completed = softimpute(observed)
    relative_error = np.linalg.norm(completed[holes] - truth[holes]) / np.linalg.norm(
        truth[holes]
    )
    assert relative_error < 0.1


def test_complete_matrix_is_returned_as_is() -> None:
    """Nothing to impute means nothing to change."""
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(6, 5))

    np.testing.assert_allclose(softimpute(matrix), matrix)


def test_larger_shrinkage_gives_a_lower_rank_estimate() -> None:
    """The shrinkage parameter controls the nuclear-norm penalty."""
    _, observed = low_rank_problem(rank=8, noise=0.5)

    light = softimpute(observed, shrinkage_value=0.5, normalize=False)
    heavy = softimpute(observed, shrinkage_value=20.0, normalize=False)

    assert np.linalg.matrix_rank(heavy, tol=1e-6) <= np.linalg.matrix_rank(
        light, tol=1e-6
    )


def test_max_rank_constrains_the_reconstruction() -> None:
    """The rank cap is a real constraint on the fit.

    With the shrinkage turned down so it does no regularizing, the rank cap is
    the only thing controlling model complexity, so capping at the true rank
    should recover the signal far better than capping at one.
    """
    rng = np.random.default_rng(8)
    true_rank = 6
    truth = rng.normal(size=(40, true_rank)) @ rng.normal(size=(true_rank, 30))
    observed = truth.copy()
    holes = rng.random(truth.shape) < 0.3
    holes[0, :] = False
    holes[:, 0] = False
    observed[holes] = np.nan

    def error(max_rank: int) -> float:
        completed = softimpute(
            observed, max_rank=max_rank, shrinkage_value=0.01, normalize=False
        )
        return float(
            np.linalg.norm(completed[holes] - truth[holes])
            / np.linalg.norm(truth[holes])
        )

    assert error(true_rank) < error(1) / 2


def test_value_clipping_is_applied() -> None:
    """min_value and max_value bound the imputed entries."""
    _, observed = low_rank_problem(rank=5, noise=0.5)
    holes = np.isnan(observed)

    completed = softimpute(observed, min_value=-1.0, max_value=1.0, normalize=False)
    assert completed[holes].min() >= -1.0 - 1e-8
    assert completed[holes].max() <= 1.0 + 1e-8


def test_normalize_flag_changes_the_result() -> None:
    """Bi-scaling is a real preprocessing step, not a no-op.

    Rows and columns are given different offsets and spreads, which is exactly
    what the bi-scaler is meant to absorb.
    """
    rng = np.random.default_rng(5)
    truth = np.outer(rng.normal(size=30), rng.normal(size=20))
    truth = truth * np.arange(1, 21)[None, :] + np.arange(30)[:, None]
    observed = truth.copy()
    holes = rng.random(truth.shape) < 0.3
    holes[0, :] = False
    holes[:, 0] = False
    observed[holes] = np.nan

    scaled = softimpute(observed, normalize=True)
    unscaled = softimpute(observed, normalize=False)

    assert not np.allclose(scaled[holes], unscaled[holes])
    # On data with strong row/column effects, bi-scaling should help.
    scaled_error = np.sqrt(np.mean((scaled[holes] - truth[holes]) ** 2))
    unscaled_error = np.sqrt(np.mean((unscaled[holes] - truth[holes]) ** 2))
    assert scaled_error < unscaled_error


def test_is_deterministic() -> None:
    """The algorithm has no randomness; repeated calls must agree exactly."""
    _, observed = low_rank_problem()

    np.testing.assert_array_equal(softimpute(observed), softimpute(observed))


def test_rejects_a_fully_missing_row() -> None:
    """A row with nothing observed cannot be imputed, and must not be faked.

    The estimator borrows strength from observed entries; with none in the row
    it would return something near the global mean, which looks like an estimate
    but carries no information.
    """
    _, observed = low_rank_problem()
    observed[3, :] = np.nan

    with pytest.raises(ValueError, match="no observed values"):
        softimpute(observed)


def test_rejects_a_fully_missing_column() -> None:
    """Same for a column with nothing observed."""
    _, observed = low_rank_problem()
    observed[:, 4] = np.nan

    with pytest.raises(ValueError, match="no observed values"):
        softimpute(observed)


def test_rejects_an_entirely_missing_matrix() -> None:
    """There is nothing to learn from."""
    with pytest.raises(ValueError):
        softimpute(np.full((4, 4), np.nan))


def test_rejects_non_matrix_input() -> None:
    """The estimator is defined for 2-dimensional data."""
    with pytest.raises(ValueError, match="2-dimensional"):
        softimpute(np.array([1.0, 2.0, np.nan]))


def test_accepts_integer_input() -> None:
    """Integer-valued ratings matrices are a normal input."""
    rng = np.random.default_rng(6)
    matrix = rng.integers(1, 6, size=(20, 15)).astype(float)
    holes = rng.random(matrix.shape) < 0.3
    holes[0, :] = False
    holes[:, 0] = False
    matrix[holes] = np.nan

    completed = softimpute(matrix)
    assert not np.any(np.isnan(completed))


def test_handles_a_non_square_matrix_either_orientation() -> None:
    """Wide and tall matrices both work; panel data is rarely square."""
    for shape in ((8, 60), (60, 8)):
        truth, observed = low_rank_problem(shape[0], shape[1], rank=2, seed=7)
        completed = softimpute(observed)
        assert completed.shape == shape
        assert not np.any(np.isnan(completed))


if __name__ == "__main__":
    pytest.main()
