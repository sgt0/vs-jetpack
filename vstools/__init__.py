"""
Functions and utils related to VapourSynth

This module is a collection of functions, utils, types, type-utils, and more
aimed at helping at having a common ground between VapourSynth packages, and
simplify writing them.
"""

from typing import Any

from .enums import *
from .exceptions import *
from .functions import *
from .types import *
from .utils import *
from .vs_proxy import *

__version__: str


def __getattr__(name: str) -> Any:
    from importlib import import_module

    if name == "__version__":
        try:
            return import_module("._version", package=__package__).__version__
        except ModuleNotFoundError:
            return "unknown"

    # TODO: add deprecation warning soon tm
    try:
        return getattr(import_module("jetpytools"), name)
    except AttributeError:
        ...

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
