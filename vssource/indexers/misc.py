from __future__ import annotations

from functools import cache
from typing import Any

from jetpytools import CustomIntEnum, DataType, SPathLike

from vstools import core, vs

from .base import Indexer

__all__ = ["FFMS2", "IMWRI", "LSMAS", "BestSource", "CarefulSource"]


_bs_msgs = set[str]()


@cache
def _add_handler_func_bs() -> None:
    from logging import WARNING, getLogger

    def handler_func_best_source(m_type: vs.MessageType, msg: str) -> None:
        if m_type == vs.MESSAGE_TYPE_INFORMATION and msg.startswith("VideoSource ") and getLogger().level <= WARNING:
            _bs_msgs.add(msg)
            print(msg, end="\r", flush=True)

    core.add_log_handler(handler_func_best_source)


class BestSource(Indexer):
    """
    BestSource indexer.
    """

    _source_func = core.lazy.bs.VideoSource

    class CacheMode(CustomIntEnum):
        """
        Cache mode.
        """

        NEVER = 0
        """
        Never read or write index to disk.
        """

        CACHE_PATH = 1
        """
        Always try to read index but only write index to disk when it will make a noticeable difference
        on subsequent runs and store index files in a subtree of *cachepath*.
        """

        CACHE_PATH_WRITE = 2
        """
        Always try to read and write index to disk and store index files in a subtree of *cachepath*.
        """

        ABSOLUTE = 3
        """
        Always try to read index but only write index to disk when it will make a noticeable difference
        on subsequent runs and store index files in the absolute path in *cachepath*
        with track number and index extension appended.
        """

        ABSOLUTE_WRITE = 4
        """
        Always try to read and write index to disk and store index files
        in the absolute path in *cachepath* with track number and index extension appended.
        """

    def __init__(self, *, force: bool = True, cachemode: CacheMode = CacheMode.ABSOLUTE, **kwargs: Any) -> None:
        _add_handler_func_bs()

        super().__init__(force=force, cachemode=cachemode, **kwargs)

    @classmethod
    def source_func(cls, path: DataType | SPathLike, *args: Any, **kwargs: Any) -> vs.VideoNode:
        _bs_msgs.clear()

        clip = super().source_func(path, *args, **kwargs)

        if _bs_msgs:
            print(flush=True)
            _bs_msgs.clear()

        return clip


class IMWRI(Indexer):
    """
    ImageMagick Writer-Reader indexer
    """

    _source_func = core.lazy.imwri.Read


class LSMAS(Indexer):
    """
    L-SMASH-Works indexer
    """

    _source_func = core.lazy.lsmas.LWLibavSource


class CarefulSource(Indexer):
    """
    CarefulSource indexer
    """

    _source_func = core.lazy.cs.ImageSource


class FFMS2(Indexer):
    """
    FFmpegSource2 indexer
    """

    _source_func = core.lazy.ffms2.Source


class ZipSource(Indexer):
    """
    vszip image reader indexer
    """

    _source_func = core.lazy.vszip.ImageRead
