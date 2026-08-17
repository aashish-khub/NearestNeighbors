"""Tests for the synthetic data-generating processes in ``nsquared.simulations``.

These were previously dead code: nothing imported the module, and both
generators carried bugs that had therefore never surfaced. The synthetic data
loader now delegates to them, so they are on the critical path and tested here.
"""

import numpy as np
import pytest

from nsquared.simulations import (
    add_gaussian_noise,
    apply_nonlinear_transform,
    combine_latent_factors,
    gendata_lin_mcar,
    gendata_nonlin_mcar,
    gendata_s_adopt,
    generate_latent_factors,
    make_mcar_mask,
    noise_scale_for_snr,
)


def test_latent_factors_have_the_requested_shape_and_support() -> None:
    """Factors are drawn uniformly from [-0.5, 0.5]."""
    row, col = generate_latent_factors(40, 25, dimensionality=6)

    assert row.shape == (40, 6)
    assert col.shape == (25, 6)
    for factors in (row, col):
        assert factors.min() >= -0.5
        assert factors.max() <= 0.5


def test_multiplicative_model_is_low_rank() -> None:
    """The bilinear signal has rank at most the latent dimension."""
    row, col = generate_latent_factors(30, 20, dimensionality=3)
    signal = combine_latent_factors(row, col, model="multiplicative")

    assert signal.shape == (30, 20)
    assert np.linalg.matrix_rank(signal, tol=1e-8) <= 3


def test_additive_model_produces_a_different_signal() -> None:
    """The Holder-continuous model is genuinely a different generator."""
    row, col = generate_latent_factors(20, 15, dimensionality=4)

    multiplicative = combine_latent_factors(row, col, model="multiplicative")
    additive = combine_latent_factors(row, col, model="additive", rho=0.5)

    assert additive.shape == multiplicative.shape
    assert not np.allclose(additive, multiplicative)


def test_unknown_combination_model_raises() -> None:
    """A typo in the model name is an error, not a silent default."""
    row, col = generate_latent_factors(5, 5)

    with pytest.raises(ValueError, match="model must be one of"):
        combine_latent_factors(row, col, model="bilinear")


@pytest.mark.parametrize("kind", ["expit", "tanh", "sin", "cubic", "sinh"])
def test_nonlinear_transforms_change_the_signal(kind: str) -> None:
    """Every supported transform actually transforms.

    The loader previously discarded the return value of its private transform
    helper, so this parameter silently did nothing.
    """
    signal = np.linspace(-2, 2, 24).reshape(6, 4)
    transformed = apply_nonlinear_transform(signal, kind)

    assert transformed.shape == signal.shape
    assert not np.allclose(transformed, signal)


def test_empty_transform_is_the_identity() -> None:
    """The default leaves the signal untouched."""
    signal = np.linspace(-2, 2, 12).reshape(3, 4)

    np.testing.assert_array_equal(apply_nonlinear_transform(signal, ""), signal)


def test_expit_does_not_overflow_on_large_inputs() -> None:
    """The logistic transform stays finite and within (0, 1)."""
    signal = np.array([[-800.0, 0.0, 800.0]])
    transformed = apply_nonlinear_transform(signal, "expit")

    assert np.all(np.isfinite(transformed))
    assert np.all((transformed >= 0.0) & (transformed <= 1.0))


def test_unknown_transform_raises() -> None:
    """An unsupported transform name is an error."""
    with pytest.raises(ValueError, match="kind must be one of"):
        apply_nonlinear_transform(np.zeros((2, 2)), "softplus")


def test_noise_scale_hits_the_requested_snr() -> None:
    """The returned standard deviation gives the target signal-to-noise ratio."""
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(200, 200))

    for snr in (0.5, 2.0, 10.0):
        stddev = noise_scale_for_snr(signal, snr)
        assert np.mean(signal**2) / stddev**2 == pytest.approx(snr)


def test_add_gaussian_noise_does_not_modify_its_input() -> None:
    """Noising returns a new array, leaving the true signal intact."""
    signal = np.ones((10, 10))
    original = signal.copy()

    noisy = add_gaussian_noise(signal, stddev=0.5)

    np.testing.assert_array_equal(signal, original)
    assert not np.allclose(noisy, signal)


def test_mcar_mask_matches_the_requested_rate() -> None:
    """Entries are dropped independently at the requested probability."""
    for miss_prob in (0.1, 0.5, 0.9):
        mask = make_mcar_mask(200, 200, miss_prob)
        assert mask.dtype == bool
        assert mask.mean() == pytest.approx(miss_prob, abs=0.02)


@pytest.mark.parametrize("generator", [gendata_lin_mcar, gendata_nonlin_mcar])
def test_theta_is_the_noiseless_signal(generator: object) -> None:
    """Theta must be the truth, not a second copy of the noisy data.

    Both generators used to bind ``Theta`` to the same array they then noised
    in place, so anything scoring against ``Theta`` was scoring against noise.
    """
    data, theta, masking = generator(N=20, T=15, p=0.7, seed=0)  # type: ignore[operator]

    assert data.shape == theta.shape == masking.shape == (20, 15)
    assert not np.array_equal(data, theta)
    # The difference is exactly the additive noise, which is small.
    assert np.abs(data - theta).max() < 0.05


def test_nonlinear_mcar_rejects_an_unknown_model() -> None:
    """The nonlinearity name is validated."""
    with pytest.raises(ValueError):
        gendata_nonlin_mcar(N=5, T=5, p=0.5, seed=0, non_lin="softplus")


def test_staggered_adoption_returns_documented_shapes() -> None:
    """gendata_s_adopt produces the tensor shapes its docstring promises.

    It previously allocated (N, T) arrays and assigned (n, d) values into them,
    so it raised ValueError and had never run.
    """
    n_rows, n_cols, n_samples, dim = 9, 6, 10, 4
    data, masking, true_mean, true_cov = gendata_s_adopt(
        N=n_rows, T=n_cols, n=n_samples, d=dim, beta=(0.5, 0.5), seed=0
    )

    assert data.shape == (n_rows, n_cols, n_samples, dim)
    assert masking.shape == (n_rows, n_cols)
    assert true_mean.shape == (n_rows, n_cols, dim)
    assert true_cov.shape == (n_rows, n_cols, dim, dim)
    assert np.all(np.isfinite(data))


def test_staggered_adoption_is_absorbing() -> None:
    """Once a unit is treated it stays treated, which is what 'staggered' means."""
    _, masking, _, _ = gendata_s_adopt(N=12, T=8, n=5, d=2, beta=(0.5, 0.5), seed=1)

    for row in masking:
        assert np.all(np.diff(row) <= 0), "a unit became observed again after adoption"


def test_staggered_adoption_never_treated_group_is_fully_observed() -> None:
    """The last third of units are the never-treated control group."""
    n_rows = 12
    _, masking, _, _ = gendata_s_adopt(N=n_rows, T=8, n=5, d=2, beta=(0.5, 0.5), seed=2)

    assert np.all(masking[2 * n_rows // 3 :] == 1)


if __name__ == "__main__":
    pytest.main()
