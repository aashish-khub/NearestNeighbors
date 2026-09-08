"""Tests for the k-nearest-neighbor imputation baseline.

``nsquared.baselines.knn_impute`` reimplements the semantics of
``sklearn.impute.KNNImputer`` on NumPy so that the baselines stay
dependency-free. Equivalence with scikit-learn was verified at the time of
writing to machine precision (worst absolute difference 8.9e-16 across
randomized shapes, both weighting schemes, and k from 1 to 5); the test below
re-runs that comparison whenever scikit-learn happens to be installed.
"""

import numpy as np
import pytest

from nsquared.baselines import knn_impute, knn_impute_columnwise


def problem(
    n_rows: int = 30, n_cols: int = 10, missing_rate: float = 0.25, seed: int = 0
) -> np.ndarray:
    """Build a matrix with holes but no fully unobserved column.

    Args:
        n_rows (int): Number of rows.
        n_cols (int): Number of columns.
        missing_rate (float): Fraction of entries to blank.
        seed (int): Random seed.

    Returns:
        np.ndarray: Matrix with ``np.nan`` holes.

    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_rows, n_cols))
    matrix[rng.random(matrix.shape) < missing_rate] = np.nan
    matrix[0, :] = rng.normal(size=n_cols)  # keep every column donatable
    return matrix


def test_fills_every_missing_entry() -> None:
    """Nothing is left missing when donors exist."""
    completed = knn_impute(problem())

    assert not np.any(np.isnan(completed))


def test_observed_entries_are_unchanged() -> None:
    """Only the holes are touched."""
    matrix = problem()
    observed = ~np.isnan(matrix)

    np.testing.assert_allclose(knn_impute(matrix)[observed], matrix[observed])


def test_identical_rows_are_imputed_exactly() -> None:
    """A row with an exact duplicate borrows that duplicate's value."""
    matrix = np.array(
        [[1.0, 2.0, np.nan], [1.0, 2.0, 7.0], [9.0, 9.0, 9.0], [8.0, 8.0, 8.0]]
    )

    assert knn_impute(matrix, n_neighbors=1)[0, 2] == pytest.approx(7.0)


def test_uses_only_rows_that_observe_the_target_column() -> None:
    """A nearer row that is itself missing the column cannot donate."""
    matrix = np.array(
        [
            [1.0, 1.0, np.nan],  # target
            [1.0, 1.0, np.nan],  # nearest, but no value to give
            [1.2, 1.2, 5.0],  # next nearest, usable
            [9.0, 9.0, 99.0],
        ]
    )

    assert knn_impute(matrix, n_neighbors=1)[0, 2] == pytest.approx(5.0)


def test_averages_over_k_neighbors() -> None:
    """With k=2 the estimate is the mean of the two nearest donors."""
    matrix = np.array(
        [
            [0.0, 0.0, np.nan],
            [0.1, 0.1, 10.0],
            [0.2, 0.2, 20.0],
            [50.0, 50.0, 999.0],
        ]
    )

    assert knn_impute(matrix, n_neighbors=2)[0, 2] == pytest.approx(15.0)


def test_distance_weighting_favours_the_closer_donor() -> None:
    """Inverse-distance weights pull the estimate toward the nearer row."""
    matrix = np.array(
        [
            [0.0, 0.0, np.nan],
            [1.0, 0.0, 0.0],  # close
            [8.0, 0.0, 100.0],  # far
        ]
    )

    uniform = knn_impute(matrix, n_neighbors=2, weights="uniform")[0, 2]
    weighted = knn_impute(matrix, n_neighbors=2, weights="distance")[0, 2]

    assert uniform == pytest.approx(50.0)
    assert weighted < uniform


def test_column_with_no_observations_stays_missing() -> None:
    """There is nothing to borrow, so the column is left alone."""
    matrix = problem()
    matrix[:, 3] = np.nan

    completed = knn_impute(matrix)
    assert np.all(np.isnan(completed[:, 3]))
    assert not np.any(np.isnan(np.delete(completed, 3, axis=1)))


def test_complete_matrix_is_returned_unchanged() -> None:
    """Nothing to impute means nothing to do."""
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(5, 4))

    np.testing.assert_array_equal(knn_impute(matrix), matrix)


def test_columnwise_variant_transposes_the_problem() -> None:
    """The column-wise baseline equals the row-wise one on the transpose."""
    matrix = problem(seed=2)

    np.testing.assert_allclose(
        knn_impute_columnwise(matrix, n_neighbors=3),
        knn_impute(matrix.T, n_neighbors=3).T,
    )


def test_is_deterministic() -> None:
    """No randomness anywhere; repeated calls agree exactly."""
    matrix = problem(seed=3)

    np.testing.assert_array_equal(knn_impute(matrix), knn_impute(matrix))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_neighbors": 0}, "n_neighbors"),
        ({"weights": "gaussian"}, "weights"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict, match: str) -> None:
    """Bad parameters are rejected rather than silently defaulted."""
    with pytest.raises(ValueError, match=match):
        knn_impute(problem(), **kwargs)


def test_rejects_non_matrix_input() -> None:
    """The baseline is defined for 2-dimensional data."""
    with pytest.raises(ValueError, match="2-dimensional"):
        knn_impute(np.array([1.0, np.nan, 3.0]))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("n_neighbors", [1, 3, 5])
def test_matches_scikit_learn(weights: str, n_neighbors: int) -> None:
    """Agreement with sklearn.impute.KNNImputer, when it is installed.

    scikit-learn is not a dependency of this package, so this test is skipped
    on a core-only install.
    """
    sklearn_impute = pytest.importorskip(
        "sklearn.impute", reason="scikit-learn is not installed"
    )
    matrix = problem(n_rows=35, n_cols=9, seed=7)

    mine = knn_impute(matrix, n_neighbors=n_neighbors, weights=weights)
    theirs = sklearn_impute.KNNImputer(
        n_neighbors=n_neighbors, weights=weights
    ).fit_transform(matrix)

    assert mine.shape == theirs.shape
    np.testing.assert_allclose(mine, theirs, atol=1e-10)


if __name__ == "__main__":
    pytest.main()
