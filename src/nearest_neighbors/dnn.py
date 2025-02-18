# Base classes for distributional nearest neighbors

import numpy as np
import numpy.typing as npt

from hyperopt import hp, Trials, fmin, tpe

from abc import ABC, abstractmethod

from .nnimputer import NNImputer

class DistNNEstimator(ABC):
    """Abstract base class for distributional nearest neighbors estimators
    """
    
    def __init__(self, 
                 nn_type = 'user-user', 
                 eta_search_space=hp.uniform('eta', 0, 1), 
                 search_algo=tpe.suggest,
                 rand_seed=None):
        """Initializes the distributional nearest neighbors estimator.
        """
        if nn_type not in ['user-user', 'item-item']:
            raise ValueError('nn_type must be one of "user-user" or "item-item".')
        
        self.nn_type = nn_type
        self.eta_search_space = eta_search_space
        self.search_algo = search_algo
        self.rand_seed = rand_seed
        self.eta = None
            
    @abstractmethod
    def distributional_distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """Returns a distance between two empirical distributions of vectors.

        Args:
            x (npt.NDArray[np.float64]): first empirical distribution
            y (npt.NDArray[np.float64]): second empirical distribution

        Returns:
            float: distance between the two empirical distributions
        """
        pass
    
    def predict(self, row_index: int, column_index: int, data: npt.NDArray[np.float64]):
        if self.eta is None:
            raise ValueError('Eta is not set. Call fit() first.')
        
        pass
        
    