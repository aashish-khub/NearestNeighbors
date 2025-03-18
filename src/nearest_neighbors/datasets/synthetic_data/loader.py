from nearest_neighbors.dataloader_base import NNDataLoader
from nearest_neighbors.dataloader_factory import register_dataset
import numpy as np
from typing import Any

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
        """Nothing to download for synthetic data"""
        print(
            "Warning: 'download' arg set to true, but there is no data to download when using synthetic data."
        )
        pass

    def _generate_simulated_data(self) -> None:
        """Generates the simulated data with no missing values"""
        # row and column latent factors:
        U = (
            np.random.uniform(size=(self.num_rows, self.latent_factor_dimensionality))
            - 0.5
        )  # U ~ Uniform(-0.5, 0.5), shape (num_rows, latent_factor_dimensionality)
        V = (
            np.random.uniform(size=(self.num_cols, self.latent_factor_dimensionality))
            - 0.5
        )  # V ~ Uniform(-0.5, 0.5), shape (num_cols, latent_factor_dimensionality)

        # Let us define a tensor Y constructed from U and V
        if self.latent_factor_combination_model == "additive":
            print("Additive model")
            # ... such that Y_i,j = |U_i + V_j|^rho * sign(U_i + V_j)
            # This Holder Continuous f(U,V), allows for a potentially non-linear relationship between U and V
            # vide: https://doi.org/10.48550/arXiv.2411.12965
            Y = np.abs(U[:, np.newaxis] + V) ** self.rho * np.sign(
                U[:, np.newaxis] + V
            )  # shape (latent_factor_dimensionality, latent_factor_dimensionality)
            Y = Y.sum(
                axis=2
            )  # collapse the latent factor dimensionality to get Y of shape (num_rows, num_cols)
        elif self.latent_factor_combination_model == "multiplicative":
            # ...such that Y_i,j = U_i * V_j
            Y = U @ V.T
        else:
            raise ValueError("Invalid latent_factor_combination_model arg provided.")
        # print("Y shape: ", Y.shape)
        self._transform_simulated_data(Y)
        data_true = Y.copy()  # i.e. "Theta"

        if (
            self.snr is not None
        ):  # if signal-to-noise ratio provided, override stddev_noise
            signal_var = np.mean(Y**2)
            stddev_noise = np.sqrt(signal_var / self.snr)
            self.stddev_noise = (
                stddev_noise  # update the instance variable for future reference
            )
        Y += self.stddev_noise * np.random.normal(size=(self.num_rows, self.num_cols))
        data_noisy = Y.copy()

        self.row_latent = U
        self.data_true = data_true
        self.data_noisy = data_noisy
        self.col_latent = V

    def _make_mcar(self) -> None:
        """Makes values missing completely at random (MCAR)"""
        # M ~ Bernoulli(self.miss_prob), shape (self.num_rows, self.num_cols)
        missing_mask = (
            np.random.binomial(1, self.miss_prob, size=(self.num_rows, self.num_cols))
            == 1
        )
        data_obs = self.data_noisy.copy()
        data_obs[missing_mask] = np.nan
        A = ~missing_mask  # A = NOT M, i.e. A_ij = 1 if Y_ij is observed, 0 if missing
        self.data_obs = data_obs
        self.availability_mask = A

    def _make_mnar(self) -> None:
        raise NotImplementedError("MNAR yet to be implemented")
        # TODO: Aashish/Caleb/Kyuesong/Tatha: come back to this later

    def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
        """Returns the full state of this object as a dictionary"""
        return_dict = {
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
        cached: bool = False,
        agg: str = "mean",  # TODO Caleb, is it safe to remove these args?
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forces a simulation run, and processes synthetic data as scalar form.

        Args:
        ----
            cached: Whether to use cached data. Default: False.
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
        self, cached: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Method not yet implemented for synthetic data"""
        # TODO Caleb, Kyuesong: is this even well-defined for synthetic data?
        raise NotImplementedError(
            "Distributional setting not yet implemented for synthetic data"
        )

    # HELPER FUNCTIONS
    def _expit(self, x: np.ndarray) -> np.ndarray:
        """Helper function to apply the logistic sigmoid function to an array"""
        return np.exp(x) / (1 + np.exp(x))

    def _transform_simulated_data(self, Y: np.ndarray) -> np.ndarray:
        """Apply a nonlinear transformation to the simulated data. Stored here since it's messy to have it in the main simulation."""
        non_lin = self.simulated_data_nonlin_transform
        if non_lin == "":
            return Y
        if non_lin == "expit":  # logistic sigmoid
            Y = self._expit(Y)
        elif non_lin == "tanh":
            Y = np.tanh(Y)
        elif non_lin == "sin":
            Y = np.sin(Y)
        elif non_lin == "cubic":
            Y = Y**3
        elif non_lin == "sinh":
            Y = np.sinh(Y)
        else:
            raise ValueError(
                "non_lin must be one of '', 'expit', 'tanh', 'sin', 'cubic', or 'sinh'."
            )
        return Y
