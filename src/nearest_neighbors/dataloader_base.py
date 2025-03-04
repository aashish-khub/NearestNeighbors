"""Abstract class for experiment/real data loading. There are two key attributes:"""
from abc import ABC, abstractmethod
import numpy as np
class NNDataLoader(ABC):
    # this is an abstract attribute to contain the URLs of the data of subclassers
    urls : dict
    def __init__(self, download:bool=False, save_dir : str="./"):
        """Initializes the data loader.
        
        Args:
            download (bool): whether to download the data locally. Default: False. If True, data is downloaded at save_dir
            save_dir (str): directory to save the data. Default: "./" (current directory). 
        """
        self.save_dir = save_dir
        self.download = download

        if download:
            # this downloads the data to save_dir . 
            self.download_data()
   
    @abstractmethod
    def download_data(self):
        """Downloads the data to self.save_dir."""
        raise NotImplementedError

    @abstractmethod
    def process_data_scalar(self, cached:bool=False) -> tuple[np.ndarray, np.ndarray]:
        """Processes the data into scalar matrix setting. Returns a tuple of ndarrays representing 2d data matrix and mask.
        
        Args:
            cached (bool): whether to use cached data. If False, then directly use urls, otherwise use self.save_dir. Default: False.
        """
        raise NotImplementedError
    
    @abstractmethod
    def process_data_distribution(self, cached:bool=False) -> tuple[np.ndarray, np.ndarray]:
        """Processes the data into distributional setting. Returns a tuple of ndarrays representing 4d distributional data and mask.
        
        Args:
            cached (bool): whether to use cached data. If False, then directly use urls, otherwise use self.save_dir. Default: False.
        """
        raise NotImplementedError
    
    