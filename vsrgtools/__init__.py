"""
Collection of various smoothing, limiting, and sharpening functions.
"""

from typing import TYPE_CHECKING, Any

from .blur import *
from .contra import *
from .enum import *
from .freqs import *
from .limit import *
from .regress import *
from .rgtools import *
from .sharp import *

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
