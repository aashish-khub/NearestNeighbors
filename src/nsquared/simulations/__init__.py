"""Data-generating processes for synthetic matrix completion experiments.

The building blocks used by
``nsquared.datasets.synthetic_data.SyntheticDataLoader``, exposed here so the
generative model can be driven directly::

    from nsquared.simulations import (
        generate_latent_factors, combine_latent_factors, add_gaussian_noise,
        make_mcar_mask,
    )

    U, V = generate_latent_factors(100, 100, dimensionality=4)
    signal = combine_latent_factors(U, V, model="multiplicative")
    observed = add_gaussian_noise(signal, stddev=0.1)
    missing = make_mcar_mask(100, 100, miss_prob=0.3)

``gendata_lin_mcar`` / ``gendata_nonlin_mcar`` are self-contained generators
that return ``(Data, Theta, Masking)`` in one call, and ``gendata_s_adopt``
generates confounded staggered-adoption data for the distributional setting.
"""

from .latent_factors import (
    COMBINATION_MODELS,
    NONLINEAR_TRANSFORMS,
    add_gaussian_noise,
    apply_nonlinear_transform,
    combine_latent_factors,
    generate_latent_factors,
    noise_scale_for_snr,
)
from .mcar import gendata_lin_mcar, gendata_nonlin_mcar, make_mcar_mask
from .mnar import gendata_s_adopt

__all__ = [
    "COMBINATION_MODELS",
    "NONLINEAR_TRANSFORMS",
    "add_gaussian_noise",
    "apply_nonlinear_transform",
    "combine_latent_factors",
    "generate_latent_factors",
    "noise_scale_for_snr",
    "make_mcar_mask",
    "gendata_lin_mcar",
    "gendata_nonlin_mcar",
    "gendata_s_adopt",
]
