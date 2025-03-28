from __future__ import annotations

import inspect

from functools import partial, wraps
from typing import Any, Callable, Literal, Union, overload

from jetpytools import P, CustomValueError, FuncExceptT, KwargsT, T

from ..enums import (
    ChromaLocation, ChromaLocationT, ColorRange, ColorRangeT, FieldBased, FieldBasedT, Matrix, MatrixT, Primaries,
    PrimariesT, PropEnum, Transfer, TransferT
)
from ..functions import DitherType, check_variable_format, depth
from ..types import ConstantFormatVideoNode, HoldsVideoFormatT, VideoFormatT, VideoNodeT
from . import vs_proxy as vs
from .cache import DynamicClipsCache
from .info import get_depth

__all__ = [
    'finalize_clip',
    'finalize_output',
    'initialize_clip',
    'initialize_input',

    'ProcessVariableClip',
    'ProcessVariableResClip',
    'ProcessVariableFormatClip',
    'ProcessVariableResFormatClip'
]


def finalize_clip(
    clip: vs.VideoNode,
    bits: VideoFormatT | HoldsVideoFormatT | int | None = 10,
    clamp_tv_range: bool | None = None,
    dither_type: DitherType = DitherType.AUTO,
    *, func: FuncExceptT | None = None
) -> ConstantFormatVideoNode:
    """
    Finalize a clip for output to the encoder.

    :param clip:            Clip to output.
    :param bits:            Bitdepth to output to.
    :param clamp_tv_range:  Whether to clamp to tv range. If None, decide based on clip properties.
    :param dither_type:     Dithering used for the bitdepth conversion.
    :param func:            Function returned for custom error handling.
                            This should only be set by VS package developers.

    :return:                Dithered down and optionally clamped clip.
    """
    from ..functions import limiter

    assert check_variable_format(clip, func or finalize_clip)

    if bits:
        clip = depth(clip, bits, dither_type=dither_type)

    if clamp_tv_range is None:
        try:
            clamp_tv_range = ColorRange.from_video(clip, strict=True).is_limited
        except Exception:
            clamp_tv_range = True

    if clamp_tv_range:
        clip = limiter(clip, tv_range=clamp_tv_range)

    return clip


@overload
def finalize_output(
    function: Callable[P, vs.VideoNode], /, *, bits: int | None = 10,
    clamp_tv_range: bool = True, dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> Callable[P, ConstantFormatVideoNode]:
    ...


@overload
def finalize_output(
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = True, dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> Callable[[Callable[P, vs.VideoNode]], Callable[P, ConstantFormatVideoNode]]:
    ...


def finalize_output(
    function: Callable[P, vs.VideoNode] | None = None,
    /, *,
    bits: int | None = 10,
    clamp_tv_range: bool = True, dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> Union[
    Callable[P, vs.VideoNode],
    Callable[[Callable[P, vs.VideoNode]], Callable[P, ConstantFormatVideoNode]]
]:
    """Decorator implementation of finalize_clip."""

    if function is None:
        return partial(finalize_output, bits=bits, clamp_tv_range=clamp_tv_range, dither_type=dither_type, func=func)

    @wraps(function)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> ConstantFormatVideoNode:
        return finalize_clip(function(*args, **kwargs), bits, clamp_tv_range, dither_type, func=func)

    return _wrapper


def initialize_clip(
    clip: vs.VideoNode, bits: int | None = None,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO, *, func: FuncExceptT | None = None
) -> ConstantFormatVideoNode:
    """
    Initialize a clip with default props.

    It is HIGHLY recommended to always use this function at the beginning of your scripts!

    :param clip:            Clip to initialize.
    :param bits:            Bits to dither to.
                            - If 0, no dithering is applied.
                            - If None, 16 if bit depth is lower than it, else leave untouched.
                            - If positive integer, dither to that bitdepth.
    :param matrix:          Matrix property to set. If None, tries to get the Matrix from existing props.
                            If no props are set or Matrix=2, guess from the video resolution.
    :param transfer:        Transfer property to set. If None, tries to get the Transfer from existing props.
                            If no props are set or Transfer=2, guess from the video resolution.
    :param primaries:       Primaries property to set. If None, tries to get the Primaries from existing props.
                            If no props are set or Primaries=2, guess from the video resolution.
    :param chroma_location: ChromaLocation prop to set. If None, tries to get the ChromaLocation from existing props.
                            If no props are set, guess from the video resolution.
    :param color_range:     ColorRange prop to set. If None, tries to get the ColorRange from existing props.
                            If no props are set, assume Limited Range.
    :param field_based:     FieldBased prop to set. If None, tries to get the FieldBased from existing props.
                            If no props are set, assume PROGRESSIVE.
    :param strict:          Whether to be strict about existing properties.
                            If True, throws an exception if certain frame properties are not found.
    :param dither_type:     Dithering used for the bitdepth conversion.
    :param func:            Function returned for custom error handling.
                            This should only be set by VS package developers.

    :return:                Clip with relevant frame properties set, and optionally dithered up to 16 bits by default.
    """

    func = func or initialize_clip

    assert check_variable_format(clip, func)

    values: list[tuple[type[PropEnum], Any]] = [
        (Matrix, matrix),
        (Transfer, transfer),
        (Primaries, primaries),
        (ChromaLocation, chroma_location),
        (ColorRange, color_range),
        (FieldBased, field_based)
    ]

    clip = PropEnum.ensure_presences(clip, [
        (cls if strict else cls.from_video(clip, False, func)) if value is None else cls.from_param(value, func)
        for cls, value in values
    ], func)

    if bits is None:
        bits = max(get_depth(clip), 16)
    elif bits <= 0:
        return clip

    return depth(clip, bits, dither_type=dither_type)


@overload
def initialize_input(
    function: Callable[P, VideoNodeT],
    /, *,
    bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> Callable[P, VideoNodeT]:
    ...


@overload
def initialize_input(
    *,
    bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExceptT | None = None
) -> Callable[[Callable[P, VideoNodeT]], Callable[P, VideoNodeT]]:
    ...


def initialize_input(
    function: Callable[P, vs.VideoNode] | None = None,
    /, *,
    bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> Union[
    Callable[P, VideoNodeT],
    Callable[[Callable[P, VideoNodeT]], Callable[P, VideoNodeT]]
]:
    """
    Decorator implementation of ``initialize_clip``
    """

    if function is None:
        return partial(
            initialize_input,
            bits=bits,
            matrix=matrix, transfer=transfer, primaries=primaries,
            chroma_location=chroma_location, color_range=color_range,
            field_based=field_based, strict=strict, dither_type=dither_type, func=func
        )

    init_args = dict[str, Any](
        bits=bits,
        matrix=matrix, transfer=transfer, primaries=primaries,
        chroma_location=chroma_location, color_range=color_range,
        field_based=field_based, strict=strict, dither_type=dither_type, func=func
    )

    @wraps(function)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> VideoNodeT:
        args_l = list(args)

        for i, obj in enumerate(args_l):
            if isinstance(obj, vs.VideoNode):
                args_l[i] = initialize_clip(obj, **init_args)
                return function(*args_l, **kwargs)  # type: ignore

        kwargs2 = kwargs.copy()

        for name, obj in kwargs2.items():
            if isinstance(obj, vs.VideoNode):
                kwargs2[name] = initialize_clip(obj, **init_args)
                return function(*args, **kwargs2)  # type: ignore

        for name, param in inspect.signature(function).parameters.items():
            if param.default is not inspect.Parameter.empty and isinstance(param.default, vs.VideoNode):
                return function(*args, **kwargs2 | {name: initialize_clip(param.default, **init_args)})  # type: ignore

        raise CustomValueError(
            'No VideoNode found in positional, keyword, nor default arguments!', func or initialize_input
        )

    return _wrapper


class ProcessVariableClip(DynamicClipsCache[T]):
    def __init__(
        self, clip: vs.VideoNode,
        out_dim: tuple[int, int] | Literal[False] | None = None,
        out_fmt: int | vs.VideoFormat | Literal[False] | None = None,
        cache_size: int = 10
    ) -> None:
        bk_args = KwargsT(length=clip.num_frames, keep=True)

        if out_dim is None:
            out_dim = (clip.width, clip.height)

        if out_fmt is None:
            out_fmt = clip.format or False

        if out_dim is not False and 0 in out_dim:
            out_dim = False

        if out_dim is False:
            bk_args.update(width=8, height=8, varsize=True)
        else:
            bk_args.update(width=out_dim[0], height=out_dim[1])

        if out_fmt is False:
            bk_args.update(format=vs.GRAY8, varformat=True)
        else:
            bk_args.update(format=out_fmt if isinstance(out_fmt, int) else out_fmt.id)

        super().__init__(cache_size)

        self.clip, self.out = clip, clip.std.BlankClip(**bk_args)

    def eval_clip(self) -> vs.VideoNode:
        if self.out.format and (0 not in (self.out.width, self.out.height)):
            try:
                return self.get_clip(self.get_key(self.clip))
            except Exception:
                ...

        return self.out.std.FrameEval(lambda n, f: self[self.get_key(f)], self.clip)

    def get_clip(self, key: T) -> vs.VideoNode:
        return self.process(self.normalize(self.clip, key))

    @classmethod
    def from_clip(
        cls,
        clip: vs.VideoNode
    ) -> vs.VideoNode:
        return cls(clip).eval_clip()

    @classmethod
    def from_func(
        cls,
        clip: vs.VideoNode,
        func: Callable[[vs.VideoNode], vs.VideoNode],
        out_dim: tuple[int, int] | Literal[False] | None = None,
        out_fmt: int | vs.VideoFormat | Literal[False] | None = None,
        cache_size: int = 10
    ) -> vs.VideoNode:
        class _inner(cls):  # type: ignore
            process = staticmethod(func)

        return _inner(clip, out_dim, out_fmt, cache_size).eval_clip()

    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> T:
        raise NotImplementedError

    def normalize(self, clip: vs.VideoNode, cast_to: T) -> vs.VideoNode:
        raise NotImplementedError

    def process(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip


class ProcessVariableResClip(ProcessVariableClip[tuple[int, int]]):
    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> tuple[int, int]:
        return (frame.width, frame.height)

    def normalize(self, clip: vs.VideoNode, cast_to: tuple[int, int]) -> vs.VideoNode:
        return clip.std.RemoveFrameProps().resize.Point(*cast_to).std.CopyFrameProps(clip)


class ProcessVariableFormatClip(ProcessVariableClip[vs.VideoFormat]):
    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> vs.VideoFormat:
        assert frame.format
        return frame.format

    def normalize(self, clip: vs.VideoNode, cast_to: vs.VideoFormat) -> vs.VideoNode:
        return clip.std.RemoveFrameProps().resize.Point(format=cast_to.id).std.CopyFrameProps(clip)


class ProcessVariableResFormatClip(ProcessVariableClip[tuple[int, int, vs.VideoFormat]]):
    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> tuple[int, int, vs.VideoFormat]:
        assert frame.format
        return (frame.width, frame.height, frame.format)

    def normalize(self, clip: vs.VideoNode, cast_to: tuple[int, int, vs.VideoFormat]) -> vs.VideoNode:
        w, h, fmt = cast_to
        return clip.std.RemoveFrameProps().resize.Point(w, h, fmt.id).std.CopyFrameProps(clip)
