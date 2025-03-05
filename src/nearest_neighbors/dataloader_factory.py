from typing import Dict, Type
from importlib import import_module
from .dataloader_base import NNDataLoader

# stores available dataset loaders
_DATASETS: Dict[str, Type[NNDataLoader]] = {}

def register_dataset(name: str):
    """Decorator to register a dataset loader."""
    def decorator(cls):
        _DATASETS[name] = cls
        return cls
    return decorator

def get_available_datasets() -> list[str]:
    """Returns the available dataset loaders."""
    return list(_DATASETS.keys())

class NNData:
    """Factory class to create dataset instances."""

    @staticmethod
    def create(dataset_name: str, download:bool=False, save_dir:str="./") -> NNDataLoader:
        """Create a dataset loader instance by name.
        
        Args:
            dataset_name: Name of the dataset
            download: Whether to download the data locally. Default: False. If True, data is downloaded at save_dir
            save_dir: Directory to save the data. Default: "./" (current directory).
        
        Returns:
            An instance of the requested dataset loader
            
        Raises:
            ValueError: If the dataset name is not registered

        """
        if dataset_name not in _DATASETS:
            try:
                import_module(f"nearest_neighbors.datasets.{dataset_name}")
            except ImportError:
                raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {get_available_datasets()}")
        
        return _DATASETS[dataset_name](download=download, save_dir=save_dir)