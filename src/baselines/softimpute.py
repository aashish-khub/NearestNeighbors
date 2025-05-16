from fancyimpute import SoftImpute, BiScaler
import numpy.typing as npt


def softimpute(X: npt.NDArray) -> npt.NDArray:
    """SoftImpute imputation method.

    Parameters
    ----------
    X : npt.NDArray
        N x T input data matrix with missing values as np.nan.

    Returns
    -------
    X_imputed : npt.NDArray
        The imputed data matrix.

    """
    # Create a SoftImpute instance with the provided arguments
    softimpute = SoftImpute(normalizer=BiScaler(), verbose=False)
    return softimpute.fit_transform(X)
