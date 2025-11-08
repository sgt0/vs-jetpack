"""
Wrappers for scaling and descaling functions.
"""

from typing import TYPE_CHECKING, Any

from .generic import *
from .helpers import *
from .mask import *
from .onnx import *
from .rescale import *
from .shaders import *
from .various import *

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
