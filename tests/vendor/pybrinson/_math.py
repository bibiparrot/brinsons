"""Internal numeric helpers: validation, aggregation, identity check.

Tolerances
----------
``WEIGHT_SUM_TOL = 1e-8`` — used to check ``|Σw − 1|``.
``RESIDUAL_TOL = 1e-9`` — used to check the per-period identity
``Σ(A + S + I) ≈ R_p − R_b`` returned by BHB / Brinson-Fachler.

These are deliberately tight: pybrinson never silently rescales weights
or absorbs a residual, because a wrong fill rule (zero-fill, drop, …)
would propagate into the identity without the reader noticing — exactly
the bug that fincore's ``residual`` field hides
(``fincore/attribution/brinson.py`` line 97-98).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .errors import AttributionError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._types import Segment

WEIGHT_SUM_TOL: float = 1e-8
MIN_PERIODS: int = 2
RESIDUAL_TOL: float = 1e-9
EMPTY_SEGMENT_TOL: float = 1e-12


def reject_non_finite(values: Sequence[float], *, name: str) -> None:
    """Raise :class:`AttributionError` if any value is ``NaN`` or ``±inf``."""
    for v in values:
        if not math.isfinite(v):
            raise AttributionError(
                f"{name} contains a non-finite value ({v!r}). "
                "pybrinson does not silently impute or drop missing data — "
                "the caller must clean inputs before calling."
            )


def validate_weights(weights: Sequence[float], *, name: str) -> None:
    """Raise unless ``weights`` are finite and sum to ``1`` within tolerance."""
    reject_non_finite(weights, name=name)
    total = math.fsum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise AttributionError(
            f"{name} weights must sum to 1 within {WEIGHT_SUM_TOL:g}; "
            f"got {total!r}. pybrinson never silently rescales — fix the input."
        )


def reject_empty_segments(segments: Sequence[Segment]) -> None:
    """Raise if any segment is "empty" — both weights below ``EMPTY_SEGMENT_TOL``.

    A segment with no members in either the portfolio or the benchmark
    contributes nothing to the attribution decomposition, but its
    ``portfolio_return`` / ``benchmark_return`` are then meaningless
    (typically a placeholder ``0.0`` or a stale value the caller forgot
    to drop). The R ``pa`` package has carried an open issue on exactly
    this failure mode since 2019
    (https://github.com/yl2/pa/issues/1 — "groups with no members").
    pybrinson refuses to silently accept it.
    """
    for s in segments:
        if (
            abs(s.portfolio_weight) < EMPTY_SEGMENT_TOL
            and abs(s.benchmark_weight) < EMPTY_SEGMENT_TOL
        ):
            raise AttributionError(
                f"segment {s.name!r} has zero portfolio and benchmark weight "
                "(empty / zero-member group). pybrinson refuses to compute "
                "an attribution effect on a segment with no economic exposure "
                "— drop it from the input or fix the upstream classification."
            )


def aggregate_return(weights: Sequence[float], returns: Sequence[float]) -> float:
    r"""Compute the weighted return :math:`\sum_i w_i r_i`.

    Uses :func:`math.fsum` for an exact, order-independent sum so that the
    identity check downstream is not polluted by accumulated rounding.
    """
    if len(weights) != len(returns):  # pragma: no cover
        # Internal invariant: callers always build weights and returns
        # from the same `segments` sequence, so lengths match by
        # construction. The guard stays as a defensive sentinel against
        # future refactors; it is not reachable from the public API.
        raise AttributionError(
            f"weights/returns length mismatch: {len(weights)} vs {len(returns)}"
        )
    return math.fsum(w * r for w, r in zip(weights, returns, strict=True))


def check_identity(
    allocation: float,
    selection: float,
    interaction: float,
    excess: float,
    *,
    method: str,
) -> None:
    """Raise if ``A + S + I`` does not match ``R_p − R_b`` within tolerance."""
    residual = (allocation + selection + interaction) - excess
    if abs(residual) > RESIDUAL_TOL:  # pragma: no cover
        # The Brinson / BF / KS formulas close this identity
        # algebraically whenever both weight vectors sum to 1. A
        # triggered failure means either a future bug in the
        # decomposition math or genuine numerical pathology — it is
        # not reachable on valid inputs.
        raise AttributionError(
            f"{method} identity violated: "
            f"A + S + I = {allocation + selection + interaction!r}, "
            f"excess = {excess!r}, residual = {residual!r}. "
            "This is either a numerical pathology or a bug — pybrinson "
            "refuses to store a silent residual."
        )
