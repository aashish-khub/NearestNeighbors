"""Classical (non-nearest-neighbor) matrix completion baselines.

Provided so that nearest neighbor methods can be compared against the
established alternatives on the same data:

- :func:`usvt` -- universal singular value thresholding (Chatterjee, 2015).
- :func:`softimpute` -- SoftImpute (Hastie et al., 2015), via ``fancyimpute``.

``usvt`` needs only NumPy. ``softimpute`` requires the optional ``fancyimpute``
dependency, so it is resolved lazily: importing this module works without it,
and only touching ``softimpute`` raises. Install it with::

    pip install "nsquared[baselines]"
"""

from typing import TYPE_CHECKING, Any

from .usvt import usvt

if TYPE_CHECKING:
    # Import for type checkers only. At runtime this is resolved by __getattr__
    # below, so that `fancyimpute` stays optional.
    from ._softimpute import softimpute

__all__ = ["usvt", "softimpute"]


def __getattr__(name: str) -> Any:
    """Resolve ``softimpute`` on first access so ``fancyimpute`` stays optional.

    Args:
        name (str): Attribute being looked up on the package.

    Raises:
        AttributeError: If the attribute is not part of the public API.
        ImportError: If ``softimpute`` is requested but ``fancyimpute`` is not
            installed.

    Returns:
        Any: The requested attribute.

    """
    if name == "softimpute":
        try:
            from ._softimpute import softimpute as softimpute_fn
        except ImportError as exc:
            raise ImportError(
                "nsquared.baselines.softimpute requires the optional 'fancyimpute' "
                "dependency, which is not installed. Install it with:\n"
                '    pip install "nsquared[baselines]"'
            ) from exc
        # Cache it so __getattr__ runs once. The implementation module is named
        # _softimpute rather than softimpute precisely so that this binding is
        # unambiguous: were they to share a name, the import system would rebind
        # this attribute to the module and callers would get a module back.
        globals()["softimpute"] = softimpute_fn
        return softimpute_fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
