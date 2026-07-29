"""Classical (non-nearest-neighbor) matrix completion baselines.

Provided so that nearest neighbor methods can be compared against the
established alternatives on the same data:

- :func:`usvt` -- universal singular value thresholding (Chatterjee, 2015).
- :func:`softimpute` -- SoftImpute (Mazumder et al., 2010; Hastie et al., 2015).

Both are implemented directly on NumPy and need no optional dependencies. Both
take a matrix with ``np.nan`` in the missing positions and return the completed
matrix, so they drop into the same evaluation loop as the NN imputers.
"""

from ._softimpute import softimpute
from .usvt import usvt

__all__ = ["softimpute", "usvt"]
