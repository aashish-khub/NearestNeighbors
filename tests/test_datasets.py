"""Tests for the N^2-Bench dataset registry and the synthetic data loader.

Only the synthetic loader is exercised end to end; the real datasets require a
network download and are covered by the scripts in ``examples/``.
"""

import numpy as np
import pytest

from nsquared import NNData, get_available_datasets

EXPECTED_DATASETS = {
    "heartsteps",
    "movielens",
    "prompteval",
    "prop99",
    "synthetic_data",
}


def test_registry_discovers_all_loaders() -> None:
    """Every shipped loader registers itself without needing a manual import."""
    available = set(get_available_datasets())
    assert EXPECTED_DATASETS <= available


def test_unknown_dataset_raises() -> None:
    """Asking for a dataset that does not exist is an error, not a silent None."""
    with pytest.raises(ValueError, match="not found"):
        NNData.create("no_such_dataset")


def test_unknown_dataset_is_not_blamed_on_a_missing_dependency() -> None:
    """A typo must not be reported as a missing optional dependency.

    The two failures need different fixes -- fix the name versus
    ``pip install "nsquared[data]"`` -- so they must not share a message.
    """
    with pytest.raises(ValueError) as excinfo:
        NNData.create("hartsteps")  # deliberate typo

    message = str(excinfo.value)
    assert "not found" in message
    assert "optional dependency" not in message


def test_unknown_dataset_does_not_pollute_the_registry() -> None:
    """A failed lookup must not leave the bad name in the help listing."""
    with pytest.raises(ValueError):
        NNData.create("definitely_not_a_dataset")

    assert "definitely_not_a_dataset" not in get_available_datasets()


def test_get_data_params_documents_parameters() -> None:
    """Loader parameters are introspectable as (type, default, description)."""
    params = NNData.get_data_params("synthetic_data")

    assert "num_rows" in params
    param_type, default, description = params["num_rows"]
    assert param_type is int
    assert default == 100
    assert description


def test_synthetic_scalar_shapes_agree() -> None:
    """process_data_scalar returns a matrix and a conforming mask."""
    loader = NNData.create("synthetic_data", num_rows=12, num_cols=15, seed=0)
    data, mask = loader.process_data_scalar()

    assert data.shape == (12, 15)
    assert mask.shape == (12, 15)
    assert set(np.unique(mask)) <= {0, 1}


def test_synthetic_data_is_reproducible() -> None:
    """The same seed produces the same matrix and mask.

    Note the construct-then-generate order: ``SyntheticDataLoader`` seeds the
    *global* NumPy RNG in its constructor rather than holding its own
    ``Generator``, so reproducibility only holds if each loader generates its
    data before the next one is built. Constructing both loaders up front and
    generating afterwards yields different matrices.
    """
    data1, mask1 = NNData.create(
        "synthetic_data", num_rows=10, num_cols=10, seed=123
    ).process_data_scalar()
    data2, mask2 = NNData.create(
        "synthetic_data", num_rows=10, num_cols=10, seed=123
    ).process_data_scalar()

    np.testing.assert_allclose(data1, data2)
    np.testing.assert_array_equal(mask1, mask2)


def test_missingness_rate_tracks_miss_prob() -> None:
    """A higher miss_prob leaves fewer observed entries."""
    sparse = NNData.create(
        "synthetic_data", num_rows=60, num_cols=60, seed=1, miss_prob=0.8
    )
    dense = NNData.create(
        "synthetic_data", num_rows=60, num_cols=60, seed=1, miss_prob=0.2
    )

    _, sparse_mask = sparse.process_data_scalar()
    _, dense_mask = dense.process_data_scalar()

    assert sparse_mask.mean() < dense_mask.mean()


def test_mnar_mode_is_not_yet_implemented() -> None:
    """The MNAR generator is declared but not implemented, and says so clearly.

    Pinning this makes the gap visible: when ``_make_mnar`` is implemented, this
    test fails and should be replaced with real coverage of the MNAR path.
    """
    loader = NNData.create(
        "synthetic_data", num_rows=20, num_cols=20, seed=2, mode="mnar"
    )

    with pytest.raises(NotImplementedError):
        loader.process_data_scalar()


def test_help_lists_available_datasets(capsys: pytest.CaptureFixture) -> None:
    """NNData.help() with no argument prints the registered loaders."""
    NNData.help()
    printed = capsys.readouterr().out

    for name in EXPECTED_DATASETS:
        assert name in printed


if __name__ == "__main__":
    pytest.main()


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_declared_params_match_constructor_defaults(name: str) -> None:
    """``NNData.help`` advertises the defaults the constructor actually uses."""
    import inspect
    from nsquared.datasets.dataloader_factory import _DATASETS

    signature = inspect.signature(_DATASETS[name].__init__)
    for param, (_, default, _) in NNData.get_data_params(name).items():
        assert param in signature.parameters, (
            f"{name}: {param!r} is not a constructor argument"
        )
        assert signature.parameters[param].default == default, (
            f"{name}: declared default for {param!r} differs from the constructor"
        )


def test_prompteval_scalar_requires_one_model_and_task() -> None:
    """Scalar mode is a single (template x example) matrix, so ambiguity is an error."""
    loader = NNData.create("prompteval")
    with pytest.raises(ValueError, match="single model and task"):
        loader.process_data_scalar()


def test_prop99_reads_local_file_from_save_dir(tmp_path) -> None:  # noqa: ANN001
    """A locally provided CSV under ``save_dir`` is used without downloading."""
    from nsquared.datasets.prop99.loader import Prop99DataLoader

    (tmp_path / "The_Tax_Burden_on_Tobacco.csv").write_text("a,b\n1,2\n")
    df = Prop99DataLoader._load_data(str(tmp_path))
    assert list(df.columns) == ["a", "b"]
