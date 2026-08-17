import pkgutil
from typing import Dict, Type
from importlib import import_module, util
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


# Subpackages that failed to import, and why. Populated by _discover_datasets so
# that a missing optional dependency can be reported as such instead of being
# mistaken for a misspelled dataset name.
_UNAVAILABLE: Dict[str, str] = {}


def _discover_datasets() -> None:
    """Import every dataset subpackage so its loader registers itself.

    Loaders register via the ``@register_dataset`` decorator, which only runs
    once the module is imported. Without this discovery step
    ``get_available_datasets`` would report only the loaders that happen to
    have been imported already. Subpackages whose optional dependencies are
    missing are recorded in ``_UNAVAILABLE`` rather than raising.
    """
    package = import_module("nsquared.datasets")
    for module_info in pkgutil.iter_modules(package.__path__):
        if not module_info.ispkg:
            continue
        try:
            import_module(f"nsquared.datasets.{module_info.name}")
        except ImportError as exc:
            # Optional dependency for this dataset is not installed.
            _UNAVAILABLE[module_info.name] = str(exc)


def _resolve_dataset(dataset_name: str) -> None:
    """Import a dataset's subpackage so its loader registers itself.

    Distinguishes "you asked for something that does not exist" from "the
    loader exists but an optional dependency is not installed", because the fix
    is completely different: a typo versus ``pip install "nsquared[data]"``.

    Args:
        dataset_name (str): The dataset the caller asked for.

    Raises:
        ValueError: If no such dataset exists, or if its loader cannot be
            imported because a dependency is missing.

    """
    if dataset_name in _DATASETS:
        return

    if util.find_spec(f"nsquared.datasets.{dataset_name}") is None:
        raise ValueError(
            f"Dataset {dataset_name!r} not found. "
            f"Available datasets: {get_available_datasets()}"
        )

    # The subpackage exists, so an ImportError here is a missing third-party
    # dependency rather than a bad dataset name.
    try:
        import_module(f"nsquared.datasets.{dataset_name}")
    except ImportError as exc:
        _UNAVAILABLE[dataset_name] = str(exc)
        raise ValueError(
            f"Dataset {dataset_name!r} is unavailable because an optional "
            f"dependency is missing ({exc}). Install the dataset extra with:\n"
            f'    pip install "nsquared[data]"'
        ) from exc


def get_available_datasets() -> list[str]:
    """Returns the names of all registered dataset loaders.

    Loaders whose optional dependencies are not installed are omitted; install
    them with ``pip install "nsquared[data]"``.
    """
    _discover_datasets()
    return sorted(_DATASETS.keys())


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
            ValueError: If the dataset name is not registered, or if its loader
                cannot be imported because an optional dependency is missing.

        """
        _resolve_dataset(dataset_name)
        return _DATASETS[dataset_name](download=download, save_dir=save_dir, **kwargs)

    @staticmethod
    def get_data_params(dataset_name: str) -> Dict[str, Tuple[Type, Any, str]]:
        """Get the custom parameters required for a dataset loader.

        Args:
            dataset_name: Name of the dataset

        Raises:
            ValueError: If the dataset name is not registered, or if its loader
                cannot be imported because an optional dependency is missing.

        """
        if dataset_name not in _DATASET_PARAMS:
            _resolve_dataset(dataset_name)
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
            if _UNAVAILABLE:
                print("\nUnavailable (optional dependency not installed):")
                for name in sorted(_UNAVAILABLE):
                    print(f"  - {name}")
                print('  Install them with: pip install "nsquared[data]"')
            print(
                "\nUse NNData.help('dataset_name') to get help for a specific dataset."
            )
            return

        try:
            _resolve_dataset(dataset_name)
        except ValueError as exc:
            print(exc)
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
