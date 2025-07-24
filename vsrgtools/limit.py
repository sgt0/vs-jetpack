from __future__ import annotations

from typing import Sequence

from vsexprtools import norm_expr
from vstools import PlanesT, fallback, scale_delta, vs

__all__ = ["limit_filter"]


def limit_filter(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    ref: vs.VideoNode | None = None,
    dark_thr: int | float | Sequence[int | float] = 1.0,
    bright_thr: int | float | Sequence[int | float] = 1.0,
    elast: float | Sequence[float] = 2.0,
    planes: PlanesT = None,
) -> vs.VideoNode:
    """
    Performs a soft-limiting between two clips to limit the difference of filtering while avoiding artifacts.

    Args:
        flt: Filtered clip.
        src: Source clip.
        ref: Reference clip, to compute the weight to be applied on filtering diff.
        dark_thr: Threshold (8-bit scale) to limit dark filtering diff.
        bright_thr: Threshold (8-bit scale) to limit bright filtering diff.
        elast: Elasticity of the soft threshold.
        planes: Planes to process. Defaults to all.

    Returns:
        Limited clip.
    """
    ref = fallback(ref, src)

    if not isinstance(dark_thr, Sequence):
        dark_thr = [dark_thr]

    if not isinstance(bright_thr, Sequence):
        bright_thr = [bright_thr]

    base_expr = "x y - DIFF! x z - abs DIFF_REF_ABS!"
    dark_expr = (
        "{dark_thr} {elast} * THR2! 1 THR2@ {dark_thr} - / SLOPE! "
        "DIFF_REF_ABS@ {dark_thr} <= x DIFF_REF_ABS@ THR2@ >= y "
        "y DIFF@ THR2@ DIFF_REF_ABS@ - * SLOPE@ * + ? ?"
    )
    bright_expr = dark_expr.replace("dark_thr", "bright_thr")

    return norm_expr(
        [flt, src, ref],
        f"{base_expr} x z < {dark_expr} {bright_expr} ?",
        dark_thr=[scale_delta(x, 8, flt) for x in dark_thr],
        bright_thr=[scale_delta(x, 8, flt) for x in bright_thr],
        elast=elast,
        planes=planes,
    )
