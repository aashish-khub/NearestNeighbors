"""Implementation of Nadaraya-Watson nearest neighbor algorithm."""

from .utils.kernels import gaussian, laplace, singular_box, box
from .nnimputer import NNImputer

import numpy as np
from hyperopt import tpe
from hyperopt import hp
from typing import Optional, Tuple

class NadarayaWatsonNN(NNImputer):
    valid_kernels = ["gaussian", "laplace", "singular_box", "box"]

    def __init__(
        self,
        kernel="gaussian",
        nn_type="ii",
        eta_axis=0,
        eta_space=hp.uniform("eta", 0, 1),
        search_algo=tpe.suggest,
        k=None,
        rand_seed=None,
    ):
        """Parameters
        ----------
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
        Z: np.ndarray,
        M: np.ndarray,
        eta: np.ndarray | Tuple[np.ndarray, np.ndarray],
        dists: np.ndarray | Tuple[np.ndarray, np.ndarray],
        inds: Optional[np.ndarray] = None,  # TODO
        cv: bool = True,
        debug: bool = False,
        ret_nn: bool = False,
    ) -> np.ndarray:
        """Estimate entries in inds using entries M = 1 and the Nadaraya-Watson estimator.

        NOTE: kernel is a radial kernel of the form kappa(||x - x'|| / eta) where eta is the bandwidth parameter

        Parameters
        ----------
        Z : np.ndarray
            The data matrix of shape (N, T, d).
        M : np.ndarray
            The missingness/treatment assignment pattern of shape (N, T).
        eta : np.ndarray | Tuple[np.ndarray, np.ndarray]
            Bandwidth parameter for the kernel
            NOTE: if a tuple is passed, then the first element is the row etas with shape (N, N) and the second element is the col etas with shape (T, T)
        dists : np.ndarray | Tuple[np.ndarray, np.ndarray]
            the row/column distances of Z
            NOTE: if a tuple is passed, then the first element is the row dists with shape (N, N) and the second element is the col dists with shape (T, T)
        inds : np.ndarray
            an array-like of indices into Z that will be estimated
        debug: bool
            boolean, whether to print debug information or not
        cv: bool
            NOTE: is this used in any of the methods?
        ret_nn : bool
            boolean, whether to return the neighbors or not

        Returns
        -------
        est : an np.array of shape (N, T, d) that consists of the estimates
            at inds.

        """
        match self.kernel:
            case "gaussian":
                K = gaussian(dists, eta)
            case "laplace":
                K = laplace(dists, eta)
            case "singular_box":
                K = singular_box(dists, eta)
            case "box":
                K = box(dists, eta)
            case _:
                raise ValueError(f"{self.kernel=} is not supported")

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

    def distances(
        self,
        Z: np.ndarray,
        M: np.ndarray,
        i: int = 0,
        t: int = 0,
        dist_type: str = "all",
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Compute the row/column-wise MSE distance

        NOTE: Copied from https://github.com/calebchin/DistributionalNearestNeighbors/blob/ddaed00cfc9d8b69e86273477d1bcc05eedb48ad/src/scalar_nn.py#L31

        Parameters
        ----------
        Z_masked : np.ndarray
            A (masked) matrix of size N x T.
        M : Optional[np.ndarray]
            A masking matrix of size N x T.
        i : int
            The column index to compute the distance for.
        t : int
            The column index to compute the distance for.
        dist_type : str
            String in ("all", "single entry", "u", "i").

        Returns
        -------
        np.ndarray
            The computed distances.

        """
        # apply mask to the original matrix -> if something is not observed, then
        # should not include in the mean calcs
        Z_cp = Z.copy()
        a, b, d = Z_cp.shape
        Z_cp[M != 1] = np.nan 
        if self.nn_type == "ii":
            Z_cp = np.swapaxes(Z_cp, 0, 1)
            M = M.T
        Z_br = Z_cp[:, None] # add extra dim for broadcast operation
        # all row dissims between pairs of rows
        if d == 1:
            dis = (Z_br - Z_cp)**2
            #print(dis.shape)
        else:
            dis = np.linalg.norm((Z_br - Z_cp)**2, axis = -1)
        
        # take mean over the sample dimension (now the 2nd dim)
        mean_row_dis = np.nanmean(dis, axis = 2)
        #print(mean_row_dis.shape)
        # overlap between every
        overlap = np.nansum(M[:, None] * M, axis = 2).astype('float64')
        zs = np.nonzero(overlap == 0)
        overlap[zs] = np.nan
        overlap = overlap[:, :, None]
        
        # rows with 0 overlap will have nan in this matrix
        # entry cannot be own neighbor, so dist is infinite
        mean_ovr = (mean_row_dis / overlap).squeeze()
        np.fill_diagonal(mean_ovr, np.inf)
        #print(mean_ovr.shape)
        return mean_ovr