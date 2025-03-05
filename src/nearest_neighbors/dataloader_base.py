"""Abstract class for experiment/real data loading. There are two key attributes:"""
from abc import ABC, abstractmethod
import numpy as np
import os

class NNDataLoader(ABC):
    # this is an abstract attribute to contain the URLs of the data of subclassers
    urls : dict
    supported_aggs = ['mean', 'sum', 'median', 'std', 'variance']
    def __init__(self, download:bool=False, save_dir : str="./", agg:str="mean", save_processed:bool=False):
        """Initializes the data loader.
        
        Args:
            download (bool): whether to download the data locally. Default: False. If True, data is downloaded at save_dir
            save_dir (str): directory to download the data to or where it already exists. Also the directory where the processed data will be. 
                Default: "./" (current directory). 
            agg (str): aggregation method to use to create scalar dataset. Default: "mean". 
            save_processed (bool): whether to save the processed data.  Default: False.

        """
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            print("Save directory not found. Creating it.")
            os.makedirs(save_dir)
        self.download = download
        self.save_processed = save_processed
        if agg not in self.supported_aggs:
            raise ValueError(f"Aggregation method {agg} not supported. Supported methods: {self.supported_aggs}")
        self.agg = agg

        if download:
            # this downloads the data to save_dir . 
            self.download_data()
   
    @abstractmethod
    def download_data(self) -> None:
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
    
    