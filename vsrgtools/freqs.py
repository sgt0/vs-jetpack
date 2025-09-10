from __future__ import annotations

from enum import auto
from typing import Any, Literal, Sequence, SupportsIndex, overload

from jetpytools import CustomValueError, SupportsString

from vsexprtools import ExprList, ExprOp, ExprVars, combine_expr
from vstools import (
    ConstantFormatVideoNode,
    CustomEnum,
    FuncExcept,
    HoldsVideoFormat,
    Planes,
    VideoFormatLike,
    VideoNodeIterableT,
    check_variable_format,
    flatten_vnodes,
    vs,
)

__all__ = ["MeanMode"]


class MeanMode(CustomEnum):
    """
    Enum of different mean for combining clips.
    """

    HARMONIC = 0
    """Harmonic mean, implemented as a Lehmer mean with p=0."""

    GEOMETRIC = 0.5
    """Geometric mean, implemented as a Lehmer mean with p=0.5."""

    ARITHMETIC = 1
    """Arithmetic mean."""

    CONTRAHARMONIC = 2
    """Contraharmonic mean, implemented as a Lehmer mean withs p=2"""

    LEHMER = 3
    """
    Lehmer mean, configurable with parameter `p`.

    Note: An odd number for `p` is preferred as it will avoid negative inputs.
    """

    MINIMUM = auto()
    """Minimum value across all clips"""

    MAXIMUM = auto()
    """Maximum value across all clips"""

    MEDIAN = auto()
    """Median value across all clips"""

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[MeanMode.LEHMER],
        *_clips: VideoNodeIterableT[vs.VideoNode],
        p: float = 3,
        planes: Planes = None,
        func: FuncExcept | None = None,
    ) -> ConstantFormatVideoNode:
        """
        Combine clips using the Lehmer mean with a configurable exponent.

        Args:
            *_clips: Input clips to combine.
            p: Exponent for the Lehmer mean calculation.
            planes: Which planes to process.
            func: An optional function to use for error handling.

        Returns:
            A new clip containing the combined frames.
        """

    @overload
    def __call__(
        self,
        *_clips: VideoNodeIterableT[vs.VideoNode],
        planes: Planes = None,
        func: FuncExcept | None = None,
    ) -> ConstantFormatVideoNode: ...

    def __call__(
        self,
        *_clips: VideoNodeIterableT[vs.VideoNode],
        planes: Planes = None,
        func: FuncExcept | None = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Applies the selected mean to one or more video clips.

        Args:
            *_clips: Input clips to combine.
            planes: Which planes to process.
            func: An optional function to use for error handling.
            **kwargs: Additional keyword arguments for certain modes.

                   - p (float): Exponent for `LEHMER` mode. Defaults to 3.
        Raises:
            CustomValueError: If there is no clip.

        Returns:
            A new clip containing the combined frames.
        """

        func = func or self.__class__

        clips = flatten_vnodes(_clips)

        assert check_variable_format(clips, func)

        if not clips:
            raise CustomValueError("There is no clip to evaluate.", func)

        if (n_clips := len(clips)) < 2:
            return clips[0]

        return self.expr(n_clips, **kwargs)(clips, planes=planes, func=func)

    def expr(
        self, n: SupportsIndex | Sequence[SupportsString] | HoldsVideoFormat | VideoFormatLike, **kwargs: Any
    ) -> ExprList:
        """
        Builds a mean expression using a specified mode.

        Args:
            n: n: Object from which to infer the variables.

        Returns:
            Mean expression.
        """
        evars = n if isinstance(n, Sequence) else ExprVars(n)
        n_len = len(evars)

        match self:
            case MeanMode.LEHMER:
                p = kwargs.pop("p", self.value)

                expr = ExprList([f"{clip} neutral - D{i}!" for i, clip in enumerate(evars)])

                for x in range(2):
                    expr.extend([[f"D{i}@ {p - x} pow" for i in range(n_len)], ExprOp.ADD * (n_len - 1), f"P{x}!"])

                expr.append("P1@ 0 = 0 P0@ P1@ / ? neutral +")

                return expr

            case MeanMode.HARMONIC | MeanMode.GEOMETRIC | MeanMode.CONTRAHARMONIC:
                return MeanMode.LEHMER.expr(n, p=self.value)
            case MeanMode.MEDIAN:
                n_op = (n_len - 1) // 2
                mean = "" if n_len % 2 else "+ 2 /"

                return ExprList(
                    (f"{' '.join(str(v for v in evars))} sort{n_len} drop{n_op} {mean} swap{n_op} drop{n_op}")
                )
            case MeanMode.ARITHMETIC:
                op = ExprOp.ADD
                kwargs.update(expr_suffix=f"{n_len} /")
            case MeanMode.MINIMUM:
                op = ExprOp.MIN
            case MeanMode.MAXIMUM:
                op = ExprOp.MAX

        return combine_expr(evars, op, **kwargs)
