"""Abstract class for experiment/real data loading. There are two key attributes:"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any
from nsquared.data_types import DataType


class NNDataLoader(ABC):
    """Base class for the N^2-Bench dataset loaders.

    Subclasses implement ``process_data_scalar``, ``process_data_distribution``
    and ``get_full_state_as_dict``, and are registered with
    :func:`nsquared.datasets.dataloader_factory.register_dataset` so that
    ``NNData.create(name)`` can build them.
    """

    # this is an abstract attribute to contain the URLs of the data of subclassers
    urls: dict
    supported_aggs = ["mean", "sum", "median", "std", "variance"]

    def __init__(
        self,
        agg: str = "mean",
        save_processed: bool = False,
        download: bool = False,
        save_dir: str = "./",
        **kwargs: Any,
    ):
        """Initializes the data loader.

        Args:
            agg (str): aggregation method to use to create scalar dataset. Default: "mean".
            save_processed (bool): whether to save the processed data.  Default: False.
            download (bool): accepted for compatibility; raw data is fetched on first
                use whenever it is not already present in ``save_dir``. Default: False.
            save_dir (str): directory the raw data files are written to. Default: "./".
            **kwargs: additional arguments to be passed to the subclass.

        """
        self.save_processed = save_processed
        self.download = download
        self.save_dir = save_dir
        if agg not in self.supported_aggs:
            raise ValueError(
                f"Aggregation method {agg} not supported. Supported methods: {self.supported_aggs}"
            )
        self.agg = agg

    @abstractmethod
    def process_data_scalar(self) -> tuple[np.ndarray, np.ndarray]:
        """Processes the data into scalar matrix setting. Returns a tuple of ndarrays representing 2d data matrix and mask.

        Returns:
            data: 2d data matrix
            mask: 2d mask matrix

        """
        raise NotImplementedError

    @abstractmethod
    def process_data_distribution(
        self, data_type: DataType | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Processes the data into distributional setting. Returns a tuple of ndarrays representing 4d distributional data and mask.

        Returns:
            data: 4d distributional data matrix
            mask: 2d mask matrix

        """
        raise NotImplementedError

    @abstractmethod
    def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
        """A helpful debugging tool: returns the full state of the data loader as a dictionary.

        Args:
            include_metadata (bool): whether to include metadata in the dictionary. Default: False.

        """
        raise NotImplementedError
