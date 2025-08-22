from __future__ import annotations

import sys
from types import UnionType
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)
from typing import cast as typing_cast

import vapoursynth as vs
from jetpytools import (
    MISSING,
    FileWasNotFoundError,
    FuncExceptT,
    MissingT,
    SPath,
    SPathLike,
    SupportsString,
    T,
    normalize_seq,
    to_arr,
)

from ..enums import PropEnum
from ..exceptions import FramePropError
from ..types import BoundVSMapValue, ConstantFormatVideoNode, HoldsPropValueT
from .cache import NodesPropsCache

__all__ = ["get_clip_filepath", "get_prop", "get_props", "merge_clip_props"]

_PropValue: TypeAlias = (
    int
    | float
    | str
    | bytes
    | vs.VideoFrame
    | vs.VideoNode
    | vs.AudioFrame
    | vs.AudioNode
    | Callable[..., Any]
    | list[int]
    | list[float]
    | list[str]
    | list[bytes]
    | list[vs.AudioFrame]
    | list[vs.AudioNode]
    | list[Callable[..., Any]]
)
DT = TypeVar("DT")
CT = TypeVar("CT")
_PropValueT = TypeVar("_PropValueT", bound=_PropValue)
_PropValueT0 = TypeVar("_PropValueT0", bound=_PropValue)
_PropValueT1 = TypeVar("_PropValueT1", bound=_PropValue)


_get_prop_cache = NodesPropsCache[vs.RawNode]()


def _normalize_types(types: type[T] | Iterable[type[T]]) -> tuple[type[T], ...]:
    norm_t = list[type[T]]()

    for tt in to_arr(types):
        t_origin = get_origin(tt)

        if t_origin is not None:
            if isinstance(tt, UnionType) or t_origin is Union:
                norm_t.extend(_normalize_types(get_args(tt)))
            else:
                norm_t.append(t_origin)
        else:
            norm_t.append(tt)

    return tuple(norm_t)


# One type signature
@overload
def get_prop(
    obj: HoldsPropValueT,
    key: str | type[PropEnum],
    t: type[_PropValueT],
    *,
    default: DT = ...,
    func: FuncExceptT | None = None,
) -> _PropValueT | DT: ...


# Tuple of two types signature
@overload
def get_prop(
    obj: HoldsPropValueT,
    key: str | type[PropEnum],
    t: tuple[type[_PropValueT], type[_PropValueT0]],
    *,
    default: DT = ...,
    func: FuncExceptT | None = None,
) -> _PropValueT | _PropValueT0 | DT: ...


# Tuple of three types signature
@overload
def get_prop(
    obj: HoldsPropValueT,
    key: str | type[PropEnum],
    t: tuple[type[_PropValueT], type[_PropValueT0], type[_PropValueT1]],
    *,
    default: DT = ...,
    func: FuncExceptT | None = None,
) -> _PropValueT | _PropValueT0 | _PropValueT1 | DT: ...


# Tuple of four types or more signature
@overload
def get_prop(
    obj: HoldsPropValueT,
    key: str | type[PropEnum],
    t: tuple[type[_PropValueT], ...],
    *,
    default: DT = ...,
    func: FuncExceptT | None = None,
) -> Any | DT: ...


# Signature when cast is specified
@overload
def get_prop(
    obj: HoldsPropValueT,
    key: str | type[PropEnum],
    t: type[_PropValueT] | tuple[type[_PropValue], ...],
    cast: Callable[[_PropValueT], CT],
    default: DT = ...,
    func: FuncExceptT | None = None,
) -> CT | DT: ...


# Signature for callable
@overload
def get_prop(
    obj: HoldsPropValueT,
    key: str | type[PropEnum],
    t: Literal["Callable"],
    *,
    default: DT = ...,
    func: FuncExceptT | None = None,
) -> Callable[..., Any] | DT: ...


# Generic signature
@overload
def get_prop(
    obj: HoldsPropValueT,
    key: str | type[PropEnum],
    t: type[Any] | tuple[type[Any], ...] | Literal["Callable"],
    cast: Callable[[Any], CT] = ...,
    default: DT = ...,
    func: FuncExceptT | None = None,
) -> Any | CT | DT: ...


def get_prop(
    obj: HoldsPropValueT,
    key: str | type[PropEnum],
    t: type[Any] | tuple[type[Any], ...] | Literal["Callable"],
    cast: Callable[[Any], CT] | MissingT = MISSING,
    default: DT | MissingT = MISSING,
    func: FuncExceptT | None = None,
) -> Any | CT | DT:
    """
    Get FrameProp ``prop`` from frame ``frame`` with expected type ``t``.

    Args:
        obj: Clip or frame containing props.
        key: Prop to get.
        t: Expected type of the prop.
        cast: Optional cast to apply to the value.
        default: Fallback value if missing or invalid.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        The property value (possibly cast).

    Raises:
        FramePropError if key is missing or wrong type and no default is provided.
    """
    func = func or get_prop

    if isinstance(obj, vs.RawNode):
        props = _get_prop_cache.get((obj, 0), MISSING)

        if props is MISSING:
            with obj.get_frame(0) as f:
                props = f.props.copy()

            _get_prop_cache[(obj, 0)] = props

    elif isinstance(obj, vs.RawFrame):
        props = obj.props
    else:
        props = obj

    resolved_key = key.prop_key if isinstance(key, type) and issubclass(key, PropEnum) else str(key)

    prop = props.get(resolved_key, MISSING)

    if prop is MISSING:
        if default is not MISSING:
            return default

        raise FramePropError(func, resolved_key, f'Key "{resolved_key}" not present in props!')

    if t == "Callable":
        if callable(prop):
            return prop
        if default is not MISSING:
            return default

        raise FramePropError(func, resolved_key)

    norm_t = _normalize_types(t)

    if isinstance(prop, norm_t):
        if cast is MISSING:
            return prop
        try:
            return cast(prop)
        except Exception:
            if default is not MISSING:
                return default
            raise FramePropError(func, resolved_key)

    if all(issubclass(ty, str) for ty in norm_t) and isinstance(prop, bytes):
        return prop.decode("utf-8")

    if default is not MISSING:
        return default

    raise FramePropError(
        func,
        resolved_key,
        f'Key "{resolved_key}" did not contain expected type: Expected "{t}" got "{type(prop)}"!',
    )


@overload
def merge_clip_props(*clips: ConstantFormatVideoNode, main_idx: int = 0) -> ConstantFormatVideoNode: ...


@overload
def merge_clip_props(*clips: vs.VideoNode, main_idx: int = 0) -> vs.VideoNode: ...


def merge_clip_props(*clips: vs.VideoNode, main_idx: int = 0) -> vs.VideoNode:
    """
    Merge frame properties from all provided clips.

    The props of the main clip (defined by main_idx) will be overwritten,
    and all other props will be added to it.

    Args:
        *clips: Clips which will be merged.
        main_idx: Index of the main clip to which all other clips props will be merged.

    Returns:
        First clip with all the frameprops of every other given clip merged into it.
    """

    if len(clips) == 1:
        return clips[0]

    def _merge_props(f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
        fdst = f[main_idx].copy()

        for i, frame in enumerate(f):
            if i == main_idx:
                continue

            fdst.props.update(frame.props)

        return fdst

    return clips[0].std.ModifyFrame(clips, _merge_props)


@overload
def get_clip_filepath(
    clip: vs.VideoNode,
    fallback: SPathLike | None = ...,
    strict: Literal[False] = ...,
    *,
    func: FuncExceptT | None = ...,
) -> SPath | None: ...


@overload
def get_clip_filepath(
    clip: vs.VideoNode, fallback: SPathLike | None = ..., strict: Literal[True] = ..., *, func: FuncExceptT | None = ...
) -> SPath: ...


@overload
def get_clip_filepath(
    clip: vs.VideoNode, fallback: SPathLike | None = ..., strict: bool = ..., *, func: FuncExceptT | None = ...
) -> SPath | None: ...


def get_clip_filepath(
    clip: vs.VideoNode, fallback: SPathLike | None = None, strict: bool = False, *, func: FuncExceptT | None = None
) -> SPath | None:
    """
    Helper function to get the file path from a clip.

    This functions checks for the `IdxFilePath` frame property.
    It also checks to ensure the file exists, and throws an error if it doesn't.

    Args:
        clip: The clip to get the file path from.
        fallback: Fallback file path to use if the `prop` is not found.
        strict: If True, will raise an error if the `prop` is not found. This makes it so the function will NEVER return
            False. Default: False.
        func: Function returned for error handling. This should only be set by VS package developers.

    Raises:
        FileWasNotFoundError: The file path was not found.
        FramePropError: The property was not found in the clip.
    """

    func = func or get_clip_filepath

    if fallback is not None and not (fallback_path := SPath(fallback)).exists() and strict:
        raise FileWasNotFoundError("Fallback file not found!", func, fallback_path.absolute())

    if not (path := get_prop(clip, "IdxFilePath", str, default=MISSING if strict else False, func=func)):
        return fallback_path if fallback is not None else None

    if not (spath := SPath(str(path))).exists() and not fallback:
        raise FileWasNotFoundError("File not found!", func, spath.absolute())

    if spath.exists():
        return spath

    if fallback is not None and fallback_path.exists():
        return fallback_path

    raise FileWasNotFoundError("File not found!", func, spath.absolute())


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue],
    *,
    func: FuncExceptT | None = None,
) -> dict[str, BoundVSMapValue]: ...


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue],
    cast: type[CT] | Callable[[BoundVSMapValue], CT],
    *,
    func: FuncExceptT | None = None,
) -> dict[str, CT]: ...


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue],
    *,
    default: DT,
    func: FuncExceptT | None = None,
) -> dict[str, BoundVSMapValue | DT]: ...


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue],
    cast: type[CT] | Callable[[BoundVSMapValue], CT],
    default: DT,
    func: FuncExceptT | None = None,
) -> dict[str, CT | DT]: ...


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: Sequence[type[BoundVSMapValue]],
    cast: Sequence[type[CT] | Callable[[BoundVSMapValue], CT]] | None = None,
    default: DT | Sequence[DT] | MissingT = ...,
    func: FuncExceptT | None = None,
) -> dict[str, Any]: ...


def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue] | Sequence[type[BoundVSMapValue]],
    cast: type[CT]
    | Callable[[BoundVSMapValue], CT]
    | Sequence[type[CT] | Callable[[BoundVSMapValue], CT]]
    | None = None,
    default: DT | Sequence[DT] | MissingT = MISSING,
    func: FuncExceptT | None = None,
) -> dict[str, Any]:
    """
    Get multiple frame properties from a clip.

    Args:
        obj: Clip or frame containing props.
        keys: List of props to get.
        t: Type of prop or list of types of props. If fewer types are provided than props, the last type will be used
            for the remaining props.
        cast: Cast value to this type, if specified.
        default: Fallback value. Can be a single value or a list of values. If a list is provided, it must be the same
            length as keys.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Dictionary mapping property names to their values. Values will be of type specified by cast if provided,
        otherwise of the type(s) specified in ``t`` or a default value if provided.
    """

    func = func or "get_props"

    if not keys:
        return {}

    t = normalize_seq(t, len(keys))
    ncast = typing_cast(list[type[CT | Callable[[BoundVSMapValue], CT]]], normalize_seq(cast, len(keys)))
    ndefault = normalize_seq(default, len(keys))

    props = dict[str, Any]()
    exceptions = list[Exception]()

    for k, t_, cast_, default_ in zip(keys, t, ncast, ndefault):
        try:
            prop = get_prop(obj, k, t_, cast_, default_, func)
        except Exception as e:
            exceptions.append(e)
        else:
            props[str(k)] = prop

    if exceptions:
        if sys.version_info >= (3, 11):
            raise ExceptionGroup("Multiple exceptions occurred!", exceptions)  # noqa: F821

        raise Exception(exceptions)

    return props
