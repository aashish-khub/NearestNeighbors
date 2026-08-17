"""Classical (non-nearest-neighbor) matrix completion baselines.

Provided so that the nearest neighbor methods can be compared against the
established alternatives on the same data:

- :func:`usvt` -- universal singular value thresholding (Chatterjee, 2015).
- :func:`softimpute` -- SoftImpute (Mazumder et al., 2010; Hastie et al., 2015).
- :func:`knn_impute` -- k-nearest-neighbor imputation, matching the semantics of
  ``sklearn.impute.KNNImputer``.
- :func:`knn_impute_columnwise` -- the same, over columns instead of rows.

All are implemented directly on NumPy and need no optional dependencies. All
take a matrix with ``np.nan`` in the missing positions and return the completed
matrix, so they drop into the same evaluation loop as the NN imputers.
"""

from ._knn import knn_impute, knn_impute_columnwise
from ._softimpute import softimpute
from .usvt import usvt

__all__ = ["knn_impute", "knn_impute_columnwise", "softimpute", "usvt"]
