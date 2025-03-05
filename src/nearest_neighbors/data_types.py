from .nnimputer import DataType
import numpy.typing as npt
from typing import Any
import numpy as np

class Scalar(DataType):
    """Data type for scalars."""

    def distance(self, obj1: float, obj2: float) -> float:
        """Calculate the distance between two scalars.

        Args:
            obj1 (float): Scalar 1
            obj2 (float): Scalar 2

        Returns:
            float: Distance between the two scalars

        """
        return (obj1 - obj2) ** 2

    def average(self, object_list: npt.NDArray[Any]) -> Any:
        """Calculate the average of a list of scalars.

        Args:
            object_list (list[Any]): List of scalars

        Returns:
            Any: Average of the scalars

        """
        return np.mean(object_list)
    
class DistributionKernelMMD(DataType):
    """Data type for distributions using Kernel MMD."""
    
    def __init__(self, kernel: str):
        """Initialize the distribution data type with a kernel.
        
        Args:
            kernel (str): Kernel to use for the MMD
        
        """
        supported_kernels = ['linear', 'square', 'exponential']
        
        if kernel not in supported_kernels:
            raise ValueError(f"Kernel {kernel} is not supported. Supported kernels are {supported_kernels}")
        
        self.kernel = kernel
        
    def distance(self, obj1: npt.NDArray, obj2: npt.NDArray) -> float:
        """Calculate the distance between two distributions using Kernel MMD.
        
        Args:
            obj1 (npt.NDArray): Distribution 1
            obj2 (npt.NDArray): Distribution 2
            
        Returns:
            float: Distance between the two distributions
        
        """
        
        # TODO: Implement this
        pass
    
    def average(self, object_list: npt.NDArray[npt.NDArray]) -> npt.NDArray:
        """Calculate the average of a list of distributions.

        Args:
            object_list (npt.NDArray[npt.NDArray]): List of distributions

        Returns:
            npt.NDArray: Average of the distributions
        
        """
        
        # TODO: Implement this
        pass