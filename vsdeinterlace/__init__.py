"""
VapourSynth deinterlacing and interlaced/telecined content helper functions.
"""

from typing import TYPE_CHECKING, Any

from .blending import *
from .enums import *
from .funcs import *
from .ivtc import *
from .qtgmc import *
from .utils import *

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
