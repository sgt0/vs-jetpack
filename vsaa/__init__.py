"""
Wrappers for anti-aliasing and scaling plugins and functions, aimed to have
compatibility with vs-kernels scalers.
"""

from typing import TYPE_CHECKING, Any

from .deinterlacers import *
from .funcs import *

__version__: str

if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name == "__version__":
            from importlib import import_module

            try:
                return import_module("._version", package=__package__).__version__
            except ModuleNotFoundError:
                return "unknown"

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
