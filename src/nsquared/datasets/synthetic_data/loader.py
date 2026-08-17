from nsquared.datasets.dataloader_base import NNDataLoader
from nsquared.datasets.dataloader_factory import register_dataset
import warnings

import numpy as np
from typing import Any
from nsquared.data_types import DataType
from nsquared.simulations import (
    add_gaussian_noise,
    apply_nonlinear_transform,
    combine_latent_factors,
    generate_latent_factors,
    make_mcar_mask,
    noise_scale_for_snr,
)

params = {
    "num_rows": (int, 100, "Number of rows in the dataset"),
    "num_cols": (int, 100, "Number of columns in the dataset"),
    "seed": (int, None, "Random seed for reproducibility"),
    "stddev_noise": (float, 1, "Standard deviation of noise"),
    "snr": (
        float,
        None,
        "Signal-to-noise ratio (defaults to None—if provided, overrides stddev_noise)",
    ),
    "mode": (str, "mcar", "Missing data mechanism ('mcar' or 'mnar')"),
    "miss_prob": (float, 0.5, "Probability of missing values (for MCAR)"),
    "mnar_deter": (
        bool,
        True,
        "Whether to use deterministic observation probabilities for MNAR",
    ),
    "latent_factor_combination_model": (
        str,
        "multiplicative",
        "Additive or multiplicative model for combining latent factors",
    ),
    "rho": (
        float,
        0.5,
        "Parameter for data generation in the nonlinear additive model. I.e. lambda in the Holder signal",
    ),
    "latent_factor_dimensionality": (int, 4, "Dimensionality of the latent factors"),
    "simulated_data_nonlin_transform": (
        str,
        "",
        "Nonlinear transformation to apply to the synthetic data, if any",
    ),
}


@register_dataset("synthetic_data", params)
class SyntheticDataLoader(NNDataLoader):
    """Data from the Heartsteps V1 study formatted into a matrix or tensor.
    To initialize with default settings, use: NNData.create("synthetic")
    """

    def __init__(
        self,
        download: bool = False,  # this is not needed for synthetic data but is kept for consistency
        save_dir: str = "./",
        agg: str = "mean",
        save_processed: bool = False,
        # above are superconstructor args
        # below are specific args
        num_rows: int = 100,
        num_cols: int = 100,
        mode: str = "mcar",
        rho: float = 0.5,
        seed: int | None = None,
        stddev_noise: float = 1,
        snr: float | None = None,
        miss_prob: float = 0.5,
        mnar_deter: bool = True,
        latent_factor_combination_model: str = "multiplicative",
        latent_factor_dimensionality: int = 4,
        simulated_data_nonlin_transform: str = "",
        **kwargs: Any,
    ):
        """Initializes the Synthetic data loader.

        Args:
        ----
            download: Whether to download the data locally. Default: False. If True, data is downloaded at save_dir
            save_dir: Directory to download the data to or where it already exists. Also the directory where the processed data will be. Default: "./" (current directory).
            agg: Aggregation method to use to create scalar dataset. Default: "mean".
            save_processed: Whether to save the processed data. Default: False.
            num_rows: Number of rows in the dataset. Default: 100.
            num_cols: Number of columns in the dataset. Default: 100.
            mode: Missing data mechanism ('mcar' or 'mnar'). Default: "mcar".
            rho: Parameter for data generation, or lambda in the Holder signal. Default: 0.5.
            seed: Random seed for reproducibility. Default: None.
            stddev_noise: Standard deviation of noise. Default: 1.
            snr: Signal-to-noise ratio. Default: None. If provided, overrides stddev_noise.
            miss_prob: Probability of missing values (for MCAR). Default: 0.5.
            mnar_deter: Whether to use deterministic observation probabilities for MNAR. Default: True.
            latent_factor_combination_model: Additive or multiplicative model for combining latent factors. Default: "multiplicative".
            latent_factor_dimensionality: Dimensionality of the latent factors. Default: 4.
            simulated_data_nonlin_transform: Nonlinear transformation to apply to the synthetic data, if any. Default: "", i.e. no transformation.
            kwargs: Additional keyword arguments.

        """
        super().__init__(
            download=download,
            save_dir=save_dir,
            agg=agg,
            save_processed=save_processed,
            **kwargs,
        )
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.mode = mode
        self.rho = rho
        self.stddev_noise = stddev_noise
        self.snr = snr  # if provided, this overrides stddev_noise
        self.miss_prob = miss_prob
        self.mnar_deter = mnar_deter
        self.latent_factor_combination_model = (
            latent_factor_combination_model  # "multiplicative" or "additive"
        )
        self.latent_factor_dimensionality = latent_factor_dimensionality
        self.simulated_data_nonlin_transform = simulated_data_nonlin_transform
        self.init_seed = seed
        if seed is not None:
            np.random.seed(
                seed=seed
            )  # instantiate random seed if provided but do it only once here

    def download_data(self) -> None:
        """Nothing to download for synthetic data."""
        warnings.warn(
            "download=True has no effect for synthetic data: it is generated "
            "locally, so there is nothing to fetch.",
            UserWarning,
            stacklevel=2,
        )

    def _generate_simulated_data(self) -> None:
        """Generates the simulated data with no missing values.

        Delegates to :mod:`nsquared.simulations` so that the generative model
        lives in one place and can be driven without going through a loader.
        """
        row_latent, col_latent = generate_latent_factors(
            self.num_rows, self.num_cols, self.latent_factor_dimensionality
        )
        signal = combine_latent_factors(
            row_latent,
            col_latent,
            model=self.latent_factor_combination_model,
            rho=self.rho,
        )
        signal = apply_nonlinear_transform(signal, self.simulated_data_nonlin_transform)
        data_true = signal.copy()  # i.e. "Theta"

        if self.snr is not None:
            # A signal-to-noise ratio, when given, overrides stddev_noise.
            self.stddev_noise = noise_scale_for_snr(signal, self.snr)
        data_noisy = add_gaussian_noise(signal, self.stddev_noise)

        self.row_latent = row_latent
        self.data_true = data_true
        self.data_noisy = data_noisy
        self.col_latent = col_latent

    def _make_mcar(self) -> None:
        """Makes values missing completely at random (MCAR)"""
        missing_mask = make_mcar_mask(self.num_rows, self.num_cols, self.miss_prob)
        data_obs = self.data_noisy.copy()
        data_obs[missing_mask] = np.nan
        # A_ij = 1 if Y_ij is observed, 0 if missing.
        self.availability_mask = ~missing_mask
        self.data_obs = data_obs

    def _make_mnar(self) -> None:
        raise NotImplementedError("MNAR yet to be implemented")
        # TODO: Aashish/Caleb/Kyuesong/Tatha: come back to this later

    def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
        """Returns the full state of this object as a dictionary"""
        # Annotated because the initializer holds only arrays, so the inferred
        # value type would be NDArray and the metadata dict added below would
        # not be assignable. Whether that is flagged depends on the numpy
        # version's stubs, so it fails on some interpreters and not others.
        return_dict: dict[str, Any] = {
            "data_observed": self.data_obs,
            "observed_entries": self.availability_mask,
            "full_data_true": self.data_true,
            "full_data_noisy": self.data_noisy,
            "row_latent": self.row_latent,
            "col_latent": self.col_latent,
        }
        if include_metadata:
            return_dict["generation_metadata"] = {
                "num_rows": self.num_rows,
                "num_cols": self.num_cols,
                "mode": self.mode,
                "stddev_noise": self.stddev_noise,
                "snr": self.snr,
                "miss_prob": self.miss_prob,
                "mnar_deter": self.mnar_deter,
                "latent_factor_combination_model": self.latent_factor_combination_model,
                "latent_factor_dimensionality": self.latent_factor_dimensionality,
                "simulated_data_nonlin_transform": self.simulated_data_nonlin_transform,
                "seed_at_generation": self.init_seed,
                "rho": self.rho,
            }
        return return_dict

    def run_simulation(self) -> None:
        """Runs the simulation and makes data missing as needed"""
        self._generate_simulated_data()
        if self.mode == "mcar":
            self._make_mcar()
        elif self.mode == "mnar":
            self._make_mnar()

    def process_data_scalar(
        self,
        agg: str = "mean",  # TODO Caleb, is it safe to remove these args?
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forces a simulation run, and processes synthetic data as scalar form.

        Args:
        ----
            agg: Aggregation method to use. Default: "mean".

        Returns:
        -------
            tuple: Tuple containing:
                - np.ndarray: Data matrix with missing values as NaN
                - np.ndarray: Binary availability mask (1 for observed, 0 for missing)

        """
        self.run_simulation()
        state_dict = self.get_full_state_as_dict(include_metadata=False)
        observed_data = state_dict["data_observed"]
        availability_mask = state_dict["observed_entries"]
        return (observed_data, availability_mask)

    def process_data_distribution(
        self, data_type: DataType | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Method not yet implemented for synthetic data

        Args:
            data_type: Data type to process. Default: None.

        """
        # TODO Caleb, Kyuesong: is this even well-defined for synthetic data?
        raise NotImplementedError(
            "Distributional setting not yet implemented for synthetic data"
        )
