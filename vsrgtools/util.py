from __future__ import annotations

from typing import Any, Sequence, TypeVar

from vstools import (
    ConstantFormatVideoNode, GenericVSFunction, PlanesT, check_variable, check_variable_format,
    join, normalize_planes, normalize_seq, plane, vs
)

from .enum import RemoveGrainMode, RepairMode

__all__ = [
    'norm_rmode_planes',
    'normalize_radius'
]

RModeT = TypeVar('RModeT', RemoveGrainMode, RepairMode)


def norm_rmode_planes(
    clip: vs.VideoNode, mode: int | RModeT | Sequence[int | RModeT], planes: PlanesT = None
) -> list[int]:
    assert check_variable(clip, norm_rmode_planes)

    modes_array = normalize_seq(mode, clip.format.num_planes)

    planes = normalize_planes(clip, planes)

    return [rep if i in planes else 0 for i, rep in enumerate(modes_array, 0)]


def normalize_radius(
    clip: vs.VideoNode,
    func: GenericVSFunction[ConstantFormatVideoNode],
    radius: Sequence[float | int] | dict[str, Sequence[float | int]],
    planes: PlanesT,
    **kwargs: Any
) -> ConstantFormatVideoNode:
    assert check_variable_format(clip, normalize_radius)

    if isinstance(radius, dict):
        name, radius = radius.popitem()
    else:
        name, radius = "radius", radius

    radius = normalize_seq(radius, clip.format.num_planes)

    planes = normalize_planes(clip, planes)

    if len(set(radius)) > 0:
        if len(planes) != 1:
            pplanes = [func(plane(clip, i), **kwargs | {name: rad, 'planes': 0}) for i, rad in enumerate(radius)]
            return join(pplanes)

        radius_i = radius[planes[0]]
    else:
        radius_i = radius[0]

    return func(clip, **kwargs | {name: radius_i, 'planes': planes})
