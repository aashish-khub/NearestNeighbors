import numpy as np
import scipy as sp
import ot

from .dnn import DistNNEstimator

def empirical_2_wasserstein(u: np.array, v: np.array):
    """
    Returns the squared 2-Wasserstein distance between two
    empirical distributions represented by arrays.
    Assumes the arrays of equal size and are sorted.
    """
    # Check that the arrays are of equal size and sorted
    assert len(u) == len(v)
    
    # Check that the arrays are sorted
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(u) and is_sorted(v)
    
    return np.sum(np.power(u - v,2)) / len(u)

def wasserstein2(sample1, sample2):
    """Compute the squared 2-Wasserstein distance between two samples.
    Inputs need to be in format (sample, dimension).
    Args:
        sample1 (np.array): 2D array of samples from the first distribution
        sample2 (np.array): 2D array of samples from the second distribution
    """
    C = ot.dist(sample1, sample2, metric='sqeuclidean')
    # Uniform weights for each empirical distribution
    a = np.ones(len(sample1)) / len(sample1)
    b = np.ones(len(sample2)) / len(sample2)

    # Compute optimal transport plan and Wasserstein distance
    P = ot.emd(a, b, C)
    return np.sum(P * C)

def dissim(list1: list[np.array], list2: list[np.array]) -> float:
    """
    Returns a dissimilarity measure between two lists of
    data arrays. Only measures dissimilarity between
    lists that are observed.
    
    If no observations overlap then returns infinity.
    Else returns the average 2-wasserstein distance.
    """
    size = 0
    distance = 0
    for ind in range(len(list1)):
        if len(list1[ind]) == 1 and len(list2[ind]):
            distance += (list1[ind][0] - list2[ind][0]) ** 2
        else:
            # distance += adjusted_emp_wasserstein2(list1[ind], list2[ind]) # unbiased
            distance += empirical_2_wasserstein(list1[ind],list2[ind]) # biased
        size += 1
    if size == 0:
        return float('inf')
    return distance / size

class DNNWasserstein(DistNNEstimator):
    def distributional_distance(self, x, y):
        if len(x) == len(y):
            return empirical_2_wasserstein(x, y)
        else:
            return wasserstein2(x, y)
        
    def average_distributions(self, distributions):
        pass
