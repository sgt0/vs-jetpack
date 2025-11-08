"""
vs-masktools aims to provide tools and functions to manage, create, and
manipulate masks in VapourSynth.
"""

from typing import TYPE_CHECKING, Any

from .abstract import *
from .details import *
from .diff import *
from .edge import *
from .edge_funcs import *
from .exceptions import *
from .hardsub import *
from .masks import *
from .morpho import *
from .spat_funcs import *
from .types import *
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
