from fancyimpute import SoftImpute, BiScaler
import numpy.typing as npt

def softimpute(X : npt.NDArray) -> npt.NDArray:
    """SoftImpute imputation method.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data with missing values as np.nan.
    kwargs : keyword arguments
        Additional arguments to pass to the SoftImpute constructor.

    Returns
    -------
    X_imputed : array-like, shape (n_samples, n_features)
        The imputed data.

    """
    # Create a SoftImpute instance with the provided arguments
    softimpute = SoftImpute(normalizer=BiScaler())
    return softimpute.fit_transform(X)