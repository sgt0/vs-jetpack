from __future__ import annotations

from vstools import core, vs

__all__ = [
    'telecine_patterns',
    'get_field_difference'
]


def telecine_patterns(clipa: vs.VideoNode, clipb: vs.VideoNode, length: int = 5) -> list[vs.VideoNode]:
    a_select = [clipa.std.SelectEvery(length, i) for i in range(length)]
    b_select = [clipb.std.SelectEvery(length, i) for i in range(length)]

    return [
        core.std.Interleave([
            (b_select if i == j else a_select)[j] for j in range(length)
        ]) for i in range(length)
    ]


def get_field_difference(clip: vs.VideoNode) -> vs.VideoNode:
    stats = clip.std.SeparateFields(tff=True).std.PlaneStats()

    return core.akarin.PropExpr(
        [clip, stats[::2], stats[1::2]], lambda: {'FieldDifference': 'y.PlaneStatsAverage z.PlaneStatsAverage - abs'}
    )