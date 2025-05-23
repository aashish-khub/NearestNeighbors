from typing import Dict, Type
from importlib import import_module
from .dataloader_base import NNDataLoader
from typing import Any, Tuple

# stores available dataset loaders
_DATASETS: Dict[str, Type[NNDataLoader]] = {}
_DATASET_PARAMS: Dict[str, Dict[str, Tuple[Type, Any, str]]] = {}


def register_dataset(name: str, params: Dict[str, Tuple[Type, Any, str]] = {}) -> Any:
    """Decorator to register a dataset loader."""

    def decorator(cls: Type[NNDataLoader]) -> Type[NNDataLoader]:
        _DATASETS[name] = cls
        if params:
            _DATASET_PARAMS[name] = params
        return cls

    return decorator


def get_available_datasets() -> list[str]:
    """Returns the available dataset loaders."""
    return list(_DATASETS.keys())


class NNData:
    """Factory class to create dataset instances."""

    @staticmethod
    def create(
        dataset_name: str, download: bool = False, save_dir: str = "./", **kwargs: Any
    ) -> NNDataLoader:
        """Create a dataset loader instance by name.

        Args:
            dataset_name: Name of the dataset
            download: Whether to download the data locally. Default: False. If True, data is downloaded at save_dir
            save_dir: Directory to save the data. Default: "./" (current directory).
            **kwargs: Additional arguments to be passed to the dataset loader.

        Returns:
            An instance of the requested dataset loader

        Raises:
            ValueError: If the dataset name is not registered

        """
        if dataset_name not in _DATASETS:
            try:
                import_module(f"nsquared.datasets.{dataset_name}")
            except ImportError:
                raise ValueError(
                    f"Dataset {dataset_name} not found. Available datasets: {get_available_datasets()}"
                )

        return _DATASETS[dataset_name](download=download, save_dir=save_dir, **kwargs)

    @staticmethod
    def get_data_params(dataset_name: str) -> Dict[str, Tuple[Type, Any, str]]:
        """Get the custom parameters required for a dataset loader.

        Args:
            dataset_name: Name of the dataset

        """
        if dataset_name not in _DATASET_PARAMS:
            try:
                import_module(f"nsquared.datasets.{dataset_name}")
            except ImportError:
                raise ValueError(
                    f"Dataset {dataset_name} not found. Available datasets: {get_available_datasets()}"
                )
        return _DATASET_PARAMS.get(dataset_name, {})

    @staticmethod
    def help(dataset_name: str = "") -> None:
        """Display help information for datasets.

        Args:
            dataset_name: Optional name of dataset to get specific help for.
                          If "", lists all available datasets.

        """
        if dataset_name == "":
            print("Available datasets:")
            for name in get_available_datasets():
                print(f"  - {name}")
            print(
                "\nUse NNData.help('dataset_name') to get help for a specific dataset."
            )
            return

        if dataset_name not in _DATASETS:
            try:
                import_module(f"nsquared.datasets.{dataset_name}")
            except ImportError:
                print(
                    f"Dataset {dataset_name} not found. Available datasets: {get_available_datasets()}"
                )
                return

        # Get the dataset class
        dataset_class = _DATASETS[dataset_name]

        # Print general information
        print(f"Help for dataset: {dataset_name}")
        print("=" * 50)
        print(dataset_class.__doc__)

        # Print common parameters
        print("\nCommon parameters:")
        print("  download: bool = False")
        print("      Whether to download the data locally")
        print("  save_dir: str = './'")
        print("      Directory to download and save data")
        print("  agg: str = 'mean'")
        print(
            "      Aggregation method (options: 'mean', 'sum', 'median', 'std', 'variance')"
        )
        print("  save_processed: bool = False")
        print("      Whether to save processed data to disk")

        # Print dataset-specific parameters from registry
        params = NNData.get_data_params(dataset_name)
        if params:
            print("\nDataset-specific parameters:")
            for name, (param_type, default, description) in params.items():
                type_name = param_type.__name__
                print(f"  {name}: {type_name} = {default}")
                print(f"      {description}")
