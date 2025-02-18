"""
Ref: https://github.com/jmetzen/kernel_regression/blob/master/kernel_regression.py
"""
from ..util_k import (
    gaussian, laplace, sobolev,
    gaussian_M, laplace_M,
    # singular, box
    box, epanechnikov,
    wendland,
)
# from .util_thin import sd_thin, kt_thin2
# from .util_thin_dnc import sd_thin_dnc, kt_thin2_dnc, kt_thin1_dnc
# from .util_k_mmd import kernel_eval, to_regression_kernel, get_kernel, gauss, laplacian, euclidean_distances
# from ..util_sample import get_Xy
# from ..rfm2.util_rfm_estimators import get_rfm_regressor

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class NadarayaWatsonBase(BaseEstimator):

    def __init__(self, kernel='epanechnikov', sigma=1, postprocess=None, M=None, **kwargs):
        assert kernel in [
            'gaussian', 
            # 'laplace', 'sobolev', 
            # 'gaussian_M', 'laplace_M', 
            # 'box', 
            'epanechnikov',
            'wendland',
            ]

        self.kernel = kernel
        # self.alpha = alpha
        self.sigma = sigma
        self.postprocess = postprocess

        self.M = M

    def fit(self, X, y, X2=None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True)
        if self.postprocess:
            y = self._validate_targets(y)

        self.X_fit_, self.y_fit_ = X, y
        # allow for different X in numerator and denominator
        self.X2_ = X if X2 is None else X2

        return self

    def predict(self, X):
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        if self.kernel == 'gaussian_M':
            # print('76> M_gauss')
            # K = gauss(
            #     self.X_fit_, # (n, rank)
            #     X @ self.W.T, # (?, rank)
            #     self.sigma
            # )
            # print('M_', self.M_)
            K = gaussian_M(self.X_fit_, X, self.M_, self.sigma)
            K2 = gaussian_M(self.X2_, X, self.M_, self.sigma)
        elif self.kernel == 'laplace_M':
            # K = laplacian(
            #     self.X_fit_, # (n, rank)
            #     X @ self.W.T, # (?, rank)
            #     self.sigma
            # )
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
        # elif self.kernel == 'singular':
        #     K = singular(self.X_fit_, X, self.sigma)
        #     K2 = singular(self.X2_, X, self.sigma)
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

        
        if self.postprocess == 'round':
            pred = np.maximum(pred, 0)
            pred = np.minimum(pred, 1)
            return np.round(pred)
        
        elif self.postprocess == 'argmax':
            max_index = np.argmax(pred, axis=1)
            one_hot = np.zeros_like(pred)
            one_hot[np.arange(len(pred)), max_index] = 1
            return one_hot
        
        elif self.postprocess == 'threshold':
            # print('thresholding')
            # only for binary classification
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            # cast pred to int
        else:
            assert self.postprocess is None

        return pred
    
    def _validate_targets(self, y):
        # y = column_or_1d(y, warn=True)
        cls = np.unique(y)
        if len(cls) < 2:
            print(y)
            # raise ValueError(
            #     "The number of classes has to be greater than one; got %d class"
            #     % len(cls)
            # )

        self.classes_ = cls

        return y
    

class NadarayaWatsonRegressor(NadarayaWatsonBase, RegressorMixin):
    pass
class NadarayaWatsonClassifier(NadarayaWatsonBase, ClassifierMixin):
    pass