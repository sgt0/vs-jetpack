import contextlib
import math
from typing import Any

import pytest

from vsexprtools import ExprOp, ExprToken, expr_func, norm_expr
from vsexprtools.util import _get_akarin_expr_version
from vstools import ColorRange, core, vs

clip_yuv_limited = ColorRange.LIMITED.apply(core.std.BlankClip(width=2, height=2, format=vs.YUV420P8))


@pytest.mark.parametrize(
    ["token", "range_in", "expected"],
    [
        (ExprToken.LumaMin, None, 16),
        (ExprToken.ChromaMin, None, 16),
        (ExprToken.LumaMax, None, 235),
        (ExprToken.ChromaMax, None, 240),
        (ExprToken.Neutral, None, 128),
        (ExprToken.RangeHalf, None, 128),
        (ExprToken.RangeSize, None, 256),
        (ExprToken.LumaRangeMin, None, 16),
        (ExprToken.ChromaRangeMin, None, 16),
        (ExprToken.LumaRangeMax, None, 235),
        (ExprToken.ChromaRangeMax, None, 240),
        (ExprToken.LumaRangeInMin, None, 16),
        (ExprToken.LumaRangeInMin, ColorRange.LIMITED, 16),
        (ExprToken.LumaRangeInMin, ColorRange.FULL, 0),
        (ExprToken.ChromaRangeInMin, None, 16),
        (ExprToken.ChromaRangeInMin, ColorRange.LIMITED, 16),
        (ExprToken.ChromaRangeInMin, ColorRange.FULL, 0),
        (ExprToken.LumaRangeInMax, None, 235),
        (ExprToken.LumaRangeInMax, ColorRange.LIMITED, 235),
        (ExprToken.LumaRangeInMax, ColorRange.FULL, 255),
        (ExprToken.ChromaRangeInMax, None, 240),
        (ExprToken.ChromaRangeInMax, ColorRange.LIMITED, 240),
        (ExprToken.ChromaRangeInMax, ColorRange.FULL, 255),
    ],
)
def test_expr_token_get_value_limited(token: ExprToken, range_in: ColorRange | None, expected: float) -> None:
    assert token.get_value(clip_yuv_limited, None, range_in) == expected


@pytest.mark.parametrize(
    ["token", "chroma", "range_in", "expected"],
    [
        (ExprToken.RangeMin, None, None, 0),
        (ExprToken.RangeMin, False, None, 0),
        (ExprToken.RangeMin, True, None, 0),
        (ExprToken.RangeMax, None, None, 255),
        (ExprToken.RangeMax, False, None, 255),
        (ExprToken.RangeMax, True, None, 255),
        (ExprToken.RangeInMin, None, None, 16),
        (ExprToken.RangeInMin, False, None, 16),
        (ExprToken.RangeInMin, True, None, 16),
        (ExprToken.RangeInMin, None, ColorRange.FULL, 0),
        (ExprToken.RangeInMin, False, ColorRange.FULL, 0),
        (ExprToken.RangeInMin, True, ColorRange.FULL, 0),
        (ExprToken.RangeInMax, None, None, 235),
        (ExprToken.RangeInMax, False, None, 235),
        (ExprToken.RangeInMax, True, None, 240),
        (ExprToken.RangeInMax, None, ColorRange.FULL, 255),
        (ExprToken.RangeInMax, False, ColorRange.FULL, 255),
        (ExprToken.RangeInMax, True, ColorRange.FULL, 255),
    ],
)
def test_expr_token_get_value_limited_with_chroma(
    token: ExprToken, chroma: bool, range_in: ColorRange | None, expected: float
) -> None:
    assert token.get_value(clip_yuv_limited, chroma, range_in) == expected


def test_expr_token_getitem() -> None:
    assert ExprToken.LumaMax[2] == "ymax_z"


def test_expr_list() -> None: ...


def test_tuple_expr_list() -> None: ...


clip_int8 = core.std.BlankClip(None, 10, 10, vs.YUV420P8, length=10, color=[128, 64, 64], keep=True)
clip_fp32 = core.std.BlankClip(None, 10, 10, vs.YUV420PS, length=10, color=[0.5, 0.25, 0.25], keep=True)


@pytest.mark.parametrize(
    ["clip", "expected"],
    [
        (clip_fp32.std.BlankClip(color=[0, 0, 0]), 0),
        (clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]), -1),
        (clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]), 1),
    ],
)
def test_expr_op_str_sgn(clip: vs.VideoNode, expected: Any) -> None:
    clip = expr_func(clip, f"x {ExprOp.SGN.convert_extra()}")

    for f in clip.frames(close=True):
        assert f[0][0, 0] == expected


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_neg(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.NEG.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == -f_in[0][0, 0]


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_tan(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.TAN.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == pytest.approx(math.tan(f_in[0][0, 0]))


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_atan(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.ATAN.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == pytest.approx(math.atan(f_in[0][0, 0]))


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_asin(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.ASIN.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == pytest.approx(math.asin(f_in[0][0, 0]))


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_acos(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.ACOS.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == pytest.approx(math.acos(f_in[0][0, 0]))


def test_expr_op_str_ceil(input_clip: vs.VideoNode = clip_fp32) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.CEIL.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == math.ceil(f_in[0][0, 0])


@pytest.mark.parametrize(
    ["clip_a", "clip_b", "t"],
    [
        (
            clip_fp32.std.BlankClip(color=[0.4376836998088198, -0.19098552065281704, 0.3494137182200806]),
            clip_fp32.std.BlankClip(color=[0.8609435397529109, -0.2693605227534943, -0.055274461226768934]),
            0.595053469921834,
        ),
        (
            clip_int8.std.BlankClip(color=[25, 196, 106]),
            clip_int8.std.BlankClip(color=[209, 58, 143]),
            0.3852455650184188,
        ),
    ],
)
@pytest.mark.parametrize("legacy", (False, True))
def test_expr_op_str_lerp(clip_a: vs.VideoNode, clip_b: vs.VideoNode, t: float, legacy: bool) -> None:
    def lerp(x: float, y: float, z: float) -> float:
        return (1 - z) * x + z * y

    if not legacy:
        _get_akarin_expr_version.cache_clear()
    else:
        with contextlib.suppress(ValueError):
            _get_akarin_expr_version()["expr_features"].remove(bytes(ExprOp.LERP.value, "utf-8"))

    expr = norm_expr([clip_a, clip_b], f"x y {t} {ExprOp.LERP.convert_extra()}")

    for f_expr, f_clip_a, f_clip_b in zip(
        expr.frames(close=True), clip_a.frames(close=True), clip_b.frames(close=True)
    ):
        for i in range(f_expr.format.num_planes):
            result = f_expr[i][0, 0]
            expected = lerp(f_clip_a[i][0, 0], f_clip_b[i][0, 0], t)

            if expr.format.sample_type == vs.INTEGER:
                assert result == round(expected)
            else:
                assert result == pytest.approx(expected, rel=1e-7)


def test_expr_op_clamp() -> None:
    assert ExprOp.clamp(c="x").to_str() == "x ExprToken.RangeMin ExprToken.RangeMax clamp"
