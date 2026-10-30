r"""Cariño multi-period linking of arithmetic attribution effects.

Method
------
Single-period attribution effects are arithmetic but returns compound
multiplicatively, so naively summing per-period effects across periods
leaves a residual. Cariño (1999) introduces a smoothing coefficient

.. math::

   k_t = \frac{\ln(1 + R_{p,t}) - \ln(1 + R_{b,t})}{R_{p,t} - R_{b,t}},
   \qquad
   K   = \frac{\ln(1 + R_p) - \ln(1 + R_b)}{R_p - R_b},

where :math:`R_p`, :math:`R_b` are the compounded portfolio and benchmark
returns. The linked allocation effect is then

.. math::

   A^{\text{linked}} = \sum_t A_t \cdot \frac{k_t}{K},

and analogously for selection and interaction. The identity

.. math::

   \sum_t (A_t + S_t + I_t) \cdot \frac{k_t}{K} = R_p - R_b

holds exactly because each per-period excess :math:`R_{p,t} - R_{b,t}`
multiplied by :math:`k_t` collapses to :math:`\ln(1 + R_{p,t}) -
\ln(1 + R_{b,t})`, and summing the logs gives :math:`K \cdot
(R_p - R_b)`.

Limit case
----------
When :math:`R_{p,t} = R_{b,t}` (or the two are within tolerance),
:math:`k_t` is computed as the limit :math:`k_t = 1 / (1 + R_{b,t})`,
i.e. :math:`d/dx[\ln(1+x)]` evaluated at :math:`R_{b,t}`. The same fix
applies to :math:`K` when the compounded returns coincide.

When :math:`R_{p,t} \le -1` or :math:`R_{b,t} \le -1` the logarithm is
undefined and an :class:`AttributionError` is raised — Cariño linking
cannot handle a total wipe-out period.

References:
----------
Primary:

- Cariño, D. R. (1999). "Combining Attribution Effects Over Time."
  *Journal of Performance Measurement*, 3(4), 5-14.
  Open-access PDF (Russell Investments author copy):
  https://www.russellinvestments.com/-/media/files/us/insights/institutions/strategic-research/combining-attribution-effects-over-time.pdf

Cross-checks:

- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 6 (linking methods).
  https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289
- Wikipedia (with citations):
  https://en.wikipedia.org/wiki/Performance_attribution#Linking_attribution_effects

The guard-rail structure (raise on ``R ≤ −1``, fall back to the limit
when the per-period excess is within tolerance) is structurally borrowed
from `ppar/utilities.py:70-98 <https://github.com/JohnDReynolds/portfolio-performance-analytics/blob/3fcf0a91b261c404e11aaac8922835437b698d2e/ppar/utilities.py#L70-L98>`_;
the underlying math is Cariño 1999.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._math import MIN_PERIODS, check_identity
from .._types import LinkedAttribution, PeriodAttribution
from ..errors import AttributionError

if TYPE_CHECKING:
    from collections.abc import Sequence

_K_DIFF_TOL = 1e-12


def _carino_k(r_p: float, r_b: float) -> float:
    """Per-period (or global) Cariño smoothing coefficient.

    Falls back to the analytic limit ``1 / (1 + r_b)`` when ``r_p`` and
    ``r_b`` are within ``1e-12``. Raises if either return is at or below
    ``-1`` (logarithm undefined).
    """
    if r_p <= -1.0 or r_b <= -1.0:  # pragma: no cover
        # Internal defensive sentinel: the public-API guard at
        # link_carino() level already rejects per-period returns ≤ -1
        # before any compounded value can reach this helper. Kept as a
        # second line of defence against future refactors.
        raise AttributionError(
            f"Cariño linking requires returns > -1 (got R_p={r_p!r}, "
            f"R_b={r_b!r}). A period that wipes out 100% or more of "
            "value cannot be log-linearised."
        )
    if abs(r_p - r_b) <= _K_DIFF_TOL:
        return 1.0 / (1.0 + r_b)
    return (math.log1p(r_p) - math.log1p(r_b)) / (r_p - r_b)


def link_carino(periods: Sequence[PeriodAttribution]) -> LinkedAttribution:
    """Link a sequence of single-period attributions via Cariño smoothing.

    Parameters
    ----------
    periods:
        Output of :func:`~pybrinson.bhb` or :func:`~pybrinson.fachler`,
        one per period, in chronological order. Must be non-empty and
        all use the same single-period method.

    Returns:
    -------
    LinkedAttribution
        ``portfolio_return`` and ``benchmark_return`` are the compounded
        single-period returns; ``excess_return = R_p - R_b``; the
        identity ``A + S + I == excess_return`` holds within ``1e-9``.
    """
    if len(periods) < MIN_PERIODS:
        raise AttributionError(
            "link_carino() requires ≥ 2 periods; for a single period the "
            "linking step is a no-op — use the PeriodAttribution result "
            "directly."
        )
    methods = {p.method for p in periods}
    if len(methods) != 1:
        raise AttributionError(
            f"link_carino() needs a single attribution method per call; "
            f"got {sorted(methods)!r}"
        )

    # Validate per-period returns before any log call: math.log1p(-1)
    # raises ValueError, so the guard must precede the compounding step.
    for p in periods:
        if p.portfolio_return <= -1.0 or p.benchmark_return <= -1.0:
            raise AttributionError(
                f"Cariño linking requires returns > -1 (period={p.period!r}, "
                f"R_p={p.portfolio_return!r}, R_b={p.benchmark_return!r}). "
                "A period that wipes out 100% or more of value cannot be "
                "log-linearised."
            )

    log_p = math.fsum(math.log1p(p.portfolio_return) for p in periods)
    log_b = math.fsum(math.log1p(p.benchmark_return) for p in periods)
    r_p_compound = math.expm1(log_p)
    r_b_compound = math.expm1(log_b)
    excess = r_p_compound - r_b_compound

    big_k = _carino_k(r_p_compound, r_b_compound)

    alloc = 0.0
    sel = 0.0
    inter = 0.0
    for p in periods:
        k_t = _carino_k(p.portfolio_return, p.benchmark_return)
        scale = k_t / big_k
        alloc += p.allocation * scale
        sel += p.selection * scale
        inter += p.interaction * scale

    check_identity(alloc, sel, inter, excess, method="Cariño")

    return LinkedAttribution(
        periods=tuple(periods),
        portfolio_return=r_p_compound,
        benchmark_return=r_b_compound,
        excess_return=excess,
        allocation=alloc,
        selection=sel,
        interaction=inter,
        method="Cariño",
    )
