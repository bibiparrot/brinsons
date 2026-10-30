"""Frozen-dataclass data model for pybrinson inputs and results.

The whole library uses plain stdlib dataclasses; no NumPy, no pandas. A
realistic attribution input is small (≤1k segments × ≤1k periods) and the
zero-dependency install is part of pybrinson's positioning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Segment:
    """One classification group (e.g. a sector) for one period.

    Parameters
    ----------
    name:
        Human-readable identifier (e.g. ``"Equities"``, ``"UK"``).
    portfolio_weight:
        Weight of the segment in the portfolio, expressed as a fraction
        (``0.30`` means 30%). The portfolio weights across all segments
        in the same period must sum to ``1`` within ``1e-8``.
    benchmark_weight:
        Weight of the segment in the benchmark, same convention.
    portfolio_return:
        Return of the portfolio's holdings inside the segment, as a
        fraction (``0.05`` means +5%).
    benchmark_return:
        Return of the benchmark's holdings inside the segment.
    parent:
        Optional immediate-parent label for hierarchical (multi-level)
        attribution. ``None`` (the default) makes the segment a root and
        preserves the v1.0 flat-classification behaviour. For hierarchies
        deeper than one level, see :func:`pybrinson.bhb`'s ``parents``
        keyword argument, which carries the parent-of-parent chain.
    local_return:
        Optional portfolio return for this segment expressed in the
        segment's local currency. Required only for currency attribution
        via :func:`pybrinson.currency_attribution`. ``None`` by default;
        BHB / Brinson-Fachler ignore it.
    currency_return:
        Optional base-currency return attributable to the segment's
        currency movement (shared by portfolio and benchmark, since both
        hold the segment in the same local currency). Required only for
        currency attribution. Under the additive Karnosky-Singer
        convention the base-currency segment return decomposes as
        ``portfolio_return = local_return + currency_return`` (up to
        cross-term, which pybrinson treats as absorbed by continuously
        compounded inputs per Karnosky & Singer 1994).
    """

    name: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    parent: str | None = None
    local_return: float | None = None
    currency_return: float | None = None


@dataclass(frozen=True, slots=True)
class SegmentAttribution:
    """Per-segment decomposition of one period's attribution result.

    For hierarchical results (v1.1+), parent rows have ``parent`` set to
    their own grandparent label (or ``None`` for a root) and ``level``
    set to their depth in the tree (root = 0). Leaf rows preserve the
    v1.0 default ``parent = None`` and ``level = 0`` when no hierarchy
    was supplied.
    """

    name: str
    allocation: float
    selection: float
    interaction: float
    parent: str | None = None
    level: int = 0
    is_leaf: bool = True


@dataclass(frozen=True, slots=True)
class PeriodAttribution:
    """Single-period attribution result.

    The identity ``allocation + selection + interaction == excess_return``
    holds exactly (within ``1e-9``) by construction — both BHB and
    Brinson-Fachler enforce it before returning.
    """

    period: str | None
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    allocation: float
    selection: float
    interaction: float
    by_segment: tuple[SegmentAttribution, ...]
    method: str


@dataclass(frozen=True, slots=True)
class CurrencySegmentAttribution:
    """Per-segment decomposition for Karnosky-Singer currency attribution.

    Four effects instead of the three Brinson effects:

    - ``market_allocation`` — local-currency market bet (BF form)
    - ``security_selection`` — within-segment local stock pick
    - ``currency_allocation`` — currency-exposure bet (BF form on ``c_i``)
    - ``interaction`` — cross-term of active weight and local active return
    """

    name: str
    market_allocation: float
    security_selection: float
    currency_allocation: float
    interaction: float


@dataclass(frozen=True, slots=True)
class CurrencyPeriodAttribution:
    """Single-period Karnosky-Singer currency attribution result.

    The identity
    ``market_allocation + security_selection + currency_allocation
    + interaction == excess_return``
    holds within ``1e-9`` by construction.
    """

    period: str | None
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    market_allocation: float
    security_selection: float
    currency_allocation: float
    interaction: float
    by_segment: tuple[CurrencySegmentAttribution, ...]
    method: str


@dataclass(frozen=True, slots=True)
class LinkedAttribution:
    """Multi-period linked attribution result.

    For ``Carino`` and ``GRAP`` linking, the identity
    ``allocation + selection + interaction == excess_return`` holds
    additively (within ``1e-9``) and ``excess_return`` is the arithmetic
    difference of compounded returns ``R_p − R_b``.

    For ``geometric`` linking, the identity is multiplicative:
    ``(1 + allocation) · (1 + selection) − 1 == excess_return`` and
    ``excess_return`` is the geometric excess
    ``(1 + R_p) / (1 + R_b) − 1``. ``interaction`` is ``0`` in this mode
    because Bacon's geometric framework folds interaction into selection.
    """

    periods: tuple[PeriodAttribution, ...]
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    allocation: float
    selection: float
    interaction: float
    method: str
