from __future__ import annotations

import vapoursynth as vs

from ..exceptions import FramesLengthError
from ..types import FrameRange, VideoNodeT, VSFunctionNoArgs
from .normalize import normalize_franges

__all__ = [
    'shift_clip', 'shift_clip_multi',

    'process_var_clip'
]


def shift_clip(clip: VideoNodeT, offset: int) -> VideoNodeT:
    """
    Shift a clip forwards or backwards by *N* frames.

    This is useful for cases where you must compare every frame of a clip
    with the frame that comes before or after the current frame,
    like for example when performing temporal operations.

    Both positive and negative integers are allowed.
    Positive values will shift a clip forward, negative will shift a clip backward.

    :param clip:            Input clip.
    :param offset:          Number of frames to offset the clip with. Negative values are allowed.
                            Positive values will shift a clip forward,
                            negative will shift a clip backward.

    :return:                Clip that has been shifted forwards or backwards by *N* frames.
    """

    if offset > clip.num_frames - 1:
        raise FramesLengthError(shift_clip, 'offset')

    if offset < 0:
        return clip[0] * abs(offset) + clip[:offset]

    if offset > 0:
        return clip[offset:] + clip[-1] * offset

    return clip


def shift_clip_multi(clip: VideoNodeT, offsets: FrameRange = (-1, 1)) -> list[VideoNodeT]:
    """
    Shift a clip forwards or backwards multiple times by a varying amount of frames.

    This will return a clip for every shifting operation performed.
    This is a convenience function that makes handling multiple shifts easier.

    Example:

    .. code-block:: python

        >>> shift_clip_multi(clip, (-3, 3))
            [VideoNode, VideoNode, VideoNode, VideoNode, VideoNode, VideoNode, VideoNode]
                -3         -2         -1          0         +1         +2         +3

    :param clip:            Input clip.
    :param offsets:         List of frame ranges for offsetting.
                            A clip will be returned for every offset.
                            Default: (-1, 1).

    :return:                A list of clips, the amount determined by the amount of offsets.
    """

    ranges = normalize_franges(offsets)

    return [shift_clip(clip, x) for x in ranges]


def process_var_clip(
    clip: vs.VideoNode, function: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode], cache_size: int | None = None
) -> vs.VideoNode:
    """
    Process variable format/resolution clips with a given function.

    This function will temporarily assert a resolution and format for a variable clip,
    run the given function, and then return a variable format clip back.

    The function must accept a VideoNode and return a VideoNode.

    :param clip:        Input clip.
    :param function:    Function that takes and returns a single VideoNode.
    :param cache_size   The maximum number of VideoNode allowed in the cache.

    :return:            Processed variable clip.
    """
    from warnings import warn

    from ..utils import ProcessVariableResFormatClip

    warn(
        "`process_var_clip` is deprecated and will be removed in a future version!\n"
        "Use `ProcessVariableResFormatClip.from_func` instead.",
        DeprecationWarning
    )

    if cache_size is not None:
        return ProcessVariableResFormatClip.from_func(clip, function, cache_size=cache_size)

    _cached_clips = dict[str, vs.VideoNode]()

    def _eval_scale(n: int, f: vs.VideoFrame) -> vs.VideoNode:
        key = f'{f.width}_{f.height}'

        if key not in _cached_clips:
            const_clip = vs.core.resize.Point(clip, f.width, f.height)

            _cached_clips[key] = function(const_clip)

        return _cached_clips[key]

    return vs.core.std.FrameEval(clip, _eval_scale, clip, clip)
