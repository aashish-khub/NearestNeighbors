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
            obj1 (npt.NDArray): (n, d) array of n samples of dimension d
            obj2 (npt.NDArray): (m, d) array of m samples of dimension d
            
        Returns:
            float: U-statistic of the squared MMD distance between the two distributions
        
        """
        
        m = obj1.shape[0]
        n = obj2.shape[0]

        assert obj1.shape[1] == obj2.shape[1]
            

        XX = np.matmul(obj1, np.transpose(obj1)) # m by m matrix with x_i^Tx_j
        YY = np.matmul(obj2, np.transpose(obj2)) # n by n matrix with y_i^Ty_j
        XY = np.matmul(obj1, np.transpose(obj2)) # m by n matrix with x_i^Ty_j  

        if self.kernel == "linear" :
            kXX, kYY, kXY = XX, YY, XY
        if self.kernel == "square" :
            kXX, kYY, kXY = (XX + np.ones( (m, m) ))**2, (YY + np.ones( (n, n) ))**2, (XY + np.ones( (m, n) ))**2
        if self.kernel == "exponential" :
            dXX_mm = np.vstack((np.diag(XX), )*m) # m*m matrix : each row is the diagonal x_i^Tx_i
            dYY_nn = np.vstack((np.diag(YY), )*n) # n*n matrix : each row is the diagonal y_i^Ty_i
            dXX_mn = np.vstack((np.diag(XX), )*n).transpose() # m*n matrix : each row is the diagonal x_i^Tx_i
            dYY_mn = np.vstack((np.diag(YY), )*m) # m*n matrix : each row is the diagonal y_i^Ty_i

            kXX = np.exp( -0.5*( dXX_mm + dXX_mm.transpose() - 2*XX ) ) 
            kYY = np.exp( -0.5*( dYY_nn + dYY_nn.transpose() - 2*YY ) )
            kXY = np.exp( -0.5*( dXX_mn + dYY_mn - 2*XY ) )
            
        val = (kXX.sum() - np.diag(kXX).sum())/(m*(m - 1)) + (kYY.sum() - np.diag(kYY).sum())/(n*(n - 1)) - 2*kXY.sum()/(n*m)
        if val < 0 : 
            val = 0

        return val
        
    def average(self, object_list: npt.NDArray) -> npt.NDArray:
        """Calculate the average of a list of distributions.

        Args:
            object_list (npt.NDArray[npt.NDArray]): List of distributions

        Returns:
            npt.NDArray: Average of the distributions
        (Returns is a mixture of vectors regardless of the kernel)
        """
        
        return object_list
        
        
