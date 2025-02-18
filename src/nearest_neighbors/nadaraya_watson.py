"""Implementation of Nadaraya-Watson nearest neighbor algorithm.
"""
from .utils.kernels import gaussian, laplace, sobolev, singular, singular_box, box
from .nnimputer import NNImputer

import numpy as np
from hyperopt import fmin, Trials, tpe
from hyperopt import hp

class NadarayaWatsonNN(NNImputer):
    valid_kernels = ['gaussian', 'laplace', 'sobolev', 'singular', 'box']

    def __init__(
        self,
        kernel="gaussian",
        nn_type="ii",
        eta_axis=0,
        eta_space=hp.uniform('eta', 0, 1),
        search_algo=tpe.suggest,
        k=None,
        rand_seed=None,
    ):
        """
        Parameters:
        -----------
        kernel : string in valid_kernels
        nn_type : string in ("ii", "uu")
            represents the type of nearest neighbors to use
            "ii" is "item-item" nn, which is column-wise
            "uu" is "user-user" nn, which is row-wise
        eta_axis : integer in [-1, 0, 1].
                   Indicates which axis to compute the eta search over. If -1, then eta search is
                   done via blocks (i.e. not row-wise or column-wise).
        eta_space : a hyperopt hp search space
                    for example: hp.uniform('eta', 0, 1). If no eta_space is given,
                    then this example will be the default search space.
        search_algo : a hyperopt algorithm
                      for example: tpe.suggest, default is tpe.suggest.
        k : integer > 1, the number of folds in k-fold cross validation over.
            If k = None (default), the LOOCV is used. 
        rand_seed : the random seed to be used for reproducible results. 
                    If None is used (default), then the system time is used (not reproducible)
        """
        if kernel not in self.valid_kernels:
            raise ValueError(
                "{} is not a valid kernel. Currently supported kernels are {}".format(
                    kernel, ", ".join(self.valid_kernels)
                )
            )
        super().__init__(
            nn_type=nn_type,
            eta_axis=eta_axis,
            eta_space=eta_space,
            search_algo=search_algo,
            k=k,
            rand_seed=rand_seed,
        )
        self.kernel = kernel

    def estimate(
        self, 
        Z, 
        M, 
        eta, 
        dists, 
        inds, 
        cv=True, 
        debug=False,
        ret_nn=False,
    ):
        """Estimate entries in inds using entries M = 1 and an eta-bandwidth.

        TODO:
        - implement kernel functions so that they take the bandwidth eta as an argument
        """

        if self.kernel == 'gaussian_M':
            K = gaussian_M(self.X_fit_, X, self.M_, self.sigma)
            K2 = gaussian_M(self.X2_, X, self.M_, self.sigma)
        elif self.kernel == 'laplace_M':
            K = laplace_M(self.X_fit_, X, self.M_, self.sigma)
            K2 = laplace_M(self.X2_, X, self.M_, self.sigma)
        elif self.kernel == 'gaussian':
            K = gaussian(self.X_fit_, X, self.sigma)
            K2 = gaussian(self.X2_, X, self.sigma)
        elif self.kernel == 'laplace':
            K = laplace(self.X_fit_, X, self.sigma)
            K2 = laplace(self.X2_, X, self.sigma)
        elif self.kernel == 'sobolev':
            K = sobolev(self.X_fit_, X)
            K2 = sobolev(self.X2_, X)
        elif self.kernel == 'box':
            K = box(self.X_fit_, X, self.sigma)
            K2 = box(self.X2_, X, self.sigma)
        elif self.kernel == 'epanechnikov':
            K = epanechnikov(self.X_fit_, X, self.sigma)
            K2 = epanechnikov(self.X2_, X, self.sigma)
        elif self.kernel == 'wendland':
            K = wendland(self.X_fit_, X, self.sigma)
            K2 = wendland(self.X2_, X, self.sigma)
        else:
            raise ValueError(f'kernel={self.kernel} is not supported')
        
        # KRR version
        # pred = (self.sol_ @ K).T

        # Nadaraya-Watson version
        # NOTE: K must be symmetric
        K2_sum = K2.sum(axis=0)
        K2_sum_nan = np.where(K2_sum == 0, np.nan, K2_sum)
        # print(K.shape, K2.shape, K2_sum.shape, self.y_fit_.shape)
        pred = (self.y_fit_ @ K) / K2_sum_nan

        # get index of nan in K along axis=0
        nan_idx = np.argwhere(np.isnan(K))
        for i, j in nan_idx:
            # set the corresponding entry of pred to the value of y at the corresponding index
            pred[j] = self.y_fit_[i]
        # if an entry of K2_sum is zero, set the corresponding entry of pred to the value of y at the corresponding index
        pred = np.where(K2_sum == 0, 0, pred)

        return pred
    