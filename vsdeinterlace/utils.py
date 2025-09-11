from __future__ import annotations

from jetpytools import FuncExcept

from vstools import ConstantFormatVideoNode, FieldBased, FieldBasedLike, check_variable, core, vs

__all__ = ["get_field_difference", "reinterlace", "reweave", "telecine_patterns", "weave"]


def telecine_patterns(
    clipa: vs.VideoNode, clipb: vs.VideoNode, length: int = 5, func: FuncExcept | None = None
) -> list[ConstantFormatVideoNode]:
    func = func or telecine_patterns

    assert check_variable(clipa, func)
    assert check_variable(clipb, func)

    a_select = [clipa.std.SelectEvery(length, i) for i in range(length)]
    b_select = [clipb.std.SelectEvery(length, i) for i in range(length)]

    return [core.std.Interleave([(b_select if i == j else a_select)[j] for j in range(length)]) for i in range(length)]


def get_field_difference(
    clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, func: FuncExcept | None = None
) -> ConstantFormatVideoNode:
    func = func or get_field_difference

    assert check_variable(clip, func)

    tff = FieldBased.from_param_or_video(tff, clip, True, func).is_tff

    stats = clip.std.SeparateFields(tff).std.PlaneStats()

    return core.akarin.PropExpr(
        [clip, stats[::2], stats[1::2]], lambda: {"FieldDifference": "y.PlaneStatsAverage z.PlaneStatsAverage - abs"}
    )


def weave(
    clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, func: FuncExcept | None = None
) -> ConstantFormatVideoNode:
    func = func or weave

    assert check_variable(clip, func)

    tff = FieldBased.from_param_or_video(tff, clip, True, func).is_tff

    return clip.std.DoubleWeave(tff)[::2]


def reweave(
    clipa: vs.VideoNode, clipb: vs.VideoNode, tff: FieldBasedLike | bool | None = None, func: FuncExcept | None = None
) -> ConstantFormatVideoNode:
    func = func or reweave
    assert check_variable(clipa, func)
    assert check_variable(clipb, func)

    return weave(core.std.Interleave([clipa, clipb]), tff, func)


def reinterlace(
    clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, func: FuncExcept | None = None
) -> ConstantFormatVideoNode:
    func = func or reinterlace

    assert check_variable(clip, func)

    tff = FieldBased.from_param_or_video(tff, clip, True, func).is_tff

    return weave(clip.std.SeparateFields(tff).std.SelectEvery(4, (0, 3)), tff, func)
