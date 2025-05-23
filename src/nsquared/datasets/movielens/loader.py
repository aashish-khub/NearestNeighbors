"""Dataset loader for the MovieLens 1M dataset.

Source: https://grouplens.org/datasets/movielens/1m/
License: CC-BY-4.0
Paper Reference for transformation implementation:
    Yihua Li, Devavrat Shah, Dogyoon Song, and Christina Lee Yu.
    "Nearest Neighbors for Matrix Estimation Interpreted as Blind Regression for Latent Variable Model."
    IEEE Transactions on Information Theory, 2020.
    https://doi.org/10.1109/TIT.2019.2950299
"""

from nsquared.datasets.dataloader_base import NNDataLoader
from nsquared.datasets.dataloader_factory import register_dataset
from nsquared.data_types import DataType

import numpy as np
import pandas as pd
from typing import Any
import logging
from joblib import Memory
import zipfile
import os
import requests
import shutil

memory = Memory(".joblib_cache", verbose=2)

logger = logging.getLogger(__name__)

params = {
    "sample_users": (
        int,
        None,
        "Number of users to sample from the dataset. By default, returns all.",
    ),
    "sample_movies": (
        int,
        None,
        "Number of movies to sample from the dataset. By default, returns all.",
    ),
    "seed": (int, None, "Random seed for reproducibility"),
}


@register_dataset("movielens", params)
class MovieLensDataLoader(NNDataLoader):
    """Data from the MovieLens study formatted into a matrix or tensor.
    To initialize with default settings, use: NNData.create("movielens").

    """

    urls = {
        "full_archive": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    }

    def __init__(
        self,
        sample_users: int | None = None,
        sample_movies: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """Initializes the MovieLens data loader.

        Args:
        ----
            sample_users: Number of users to sample from the dataset. Default: None (use all users).
            sample_movies: Number of movies to sample from the dataset. Default: None (use all movies).
            seed: Random seed for reproducibility. Default: None
            kwargs: Additional keyword arguments.

        """
        super().__init__(
            **kwargs,
        )
        self.sample_users = sample_users
        self.sample_movies = sample_movies
        if seed is not None:
            np.random.seed(
                seed=seed
            )  # instantiate random seed if provided but do it only once here

    def process_data_scalar(self, agg: str = "mean") -> tuple[np.ndarray, np.ndarray]:
        """Process the data into scalar setting. Note that this implementation is specific to MovieLens as it calls upon functions that do specific MovieLens data processing.

        Args:
        ----
            agg: Aggregation method to use. Default: "mean". Options: "mean", "sum", "median", "std", "variance"

        Returns:
        -------
            data: 2d processed data matrix of floats
            mask: Mask for processed data

        """
        df_movies, df_ratings, df_users = self._load_data()
        data_df = df_ratings.pivot(
            index="UserID", columns="MovieID", values="Rating"
        )  # rows: UserID, columns: MovieID, values: Rating
        if self.sample_users is not None:
            data_df = data_df.sample(n=self.sample_users)
        if self.sample_movies is not None:
            data_df = data_df.sample(n=self.sample_movies, axis=1)
        mask_df = data_df.notna().astype(bool)
        data = data_df.to_numpy()
        mask = mask_df.to_numpy()
        self.data = data
        self.mask = mask
        return data, mask

    def process_data_distribution(
        self, data_type: DataType | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Does not apply to MovieLens as it is not a distributional dataset.

        Args:
            data_type: Data type to process. Default: None.

        """
        raise (ValueError("There is no distributional data for MovieLens."))

    def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
        """Returns the full state as a dictionary. For HeartSteps, this includes the data, masking matrix, and the custom parameters (if include_metadata == True

        If the data and mask are None, then the data has not been processed yet. Call process_data_scalar() or process_data_distribution() to process the data first.

        Args:
            include_metadata (bool): Whether to include metadata in the dictionary. Default: False. The metadata for HeartSteps is currently empty.

        """
        full_state = {
            "data": self.data,
            "mask": self.mask,
            "custom_params": {
                "sample_users": self.sample_users,
                "sample_movies": self.sample_movies,
            },
        }
        return full_state

    @classmethod
    @memory.cache
    def _load_data(cls) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download and extract the MovieLens 1M dataset and load its contents."""
        zip_path = "movielens_1m.zip"
        extract_dir = "movielens_1m"

        # Download if missing or previously corrupted
        needs_download = not os.path.exists(zip_path)
        if not needs_download:
            try:
                with zipfile.ZipFile(zip_path, "r") as test_zip:
                    test_zip.testzip()
            except zipfile.BadZipFile:
                os.remove(zip_path)
                needs_download = True

        if needs_download:
            logger.info("Downloading MovieLens 1M dataset...")
            response = requests.get(cls.urls["full_archive"], stream=True)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to download dataset: {response.status_code}"
                )
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)

        if not os.path.exists(extract_dir):
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        base = os.path.join(extract_dir, "ml-1m")
        df_movies = pd.read_csv(
            os.path.join(base, "movies.dat"),
            sep="::",
            header=None,
            engine="python",
            encoding="ISO-8859-1",
            names=["MovieID", "Title", "Genres"],
        )
        df_ratings = pd.read_csv(
            os.path.join(base, "ratings.dat"),
            sep="::",
            header=None,
            engine="python",
            encoding="ISO-8859-1",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
        )
        df_users = pd.read_csv(
            os.path.join(base, "users.dat"),
            sep="::",
            header=None,
            engine="python",
            encoding="ISO-8859-1",
            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        )
        # delete the zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return df_movies, df_ratings, df_users
