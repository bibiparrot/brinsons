r"""Bacon-style geometric multi-period linking.

Method
------
The arithmetic Brinson-Hood-Beebower decomposition can be rewritten in a
geometric form (Bacon 2008, chap. 6) where each period's allocation and
selection effects are themselves *returns*, and they compound naturally
across periods with **no smoothing residual** by construction.

Define the per-period semi-notional return

.. math::

   R_{s,t} = \sum_i w_{p,i,t} \, R_{b,i,t}.

For BHB inputs, :math:`R_{s,t} = R_{b,t} + A_t` where :math:`A_t` is the
arithmetic BHB allocation total — so the semi-notional return is
recoverable from a :class:`~pybrinson.PeriodAttribution` alone, no
segment-level recomputation needed.

Per-period geometric effects:

.. math::

   A_t^{\text{geom}} &= \frac{R_{s,t} - R_{b,t}}{1 + R_{b,t}}
                      = \frac{A_t}{1 + R_{b,t}}, \\
   S_t^{\text{geom}} &= \frac{R_{p,t} - R_{s,t}}{1 + R_{s,t}}
                      = \frac{S_t + I_t}{1 + R_{b,t} + A_t}.

(In Bacon's geometric form the arithmetic interaction is folded into
selection. The 3-effect arithmetic decomposition has no clean geometric
analogue, so :func:`link_geometric` reports ``interaction = 0`` and the
combined arithmetic ``selection + interaction`` is what becomes
:math:`S^{\text{geom}}`.)

Multi-period linking is then trivially multiplicative:

.. math::

   1 + A^{\text{linked}} &= \prod_t \big(1 + A_t^{\text{geom}}\big), \\
   1 + S^{\text{linked}} &= \prod_t \big(1 + S_t^{\text{geom}}\big).

Identity (multiplicative, exact)
--------------------------------

.. math::

   (1 + A^{\text{linked}}) \cdot (1 + S^{\text{linked}})
     = \prod_t \frac{1 + R_{s,t}}{1 + R_{b,t}}
       \cdot \frac{1 + R_{p,t}}{1 + R_{s,t}}
     = \frac{\prod_t (1 + R_{p,t})}{\prod_t (1 + R_{b,t})}
     = \frac{1 + R_p}{1 + R_b}.

So the linked excess return reported by :func:`link_geometric` is the
**geometric** excess :math:`(1 + R_p) / (1 + R_b) - 1`, *not* the
arithmetic difference :math:`R_p - R_b`. This is the only linking method
in pybrinson that uses a multiplicative identity; Cariño and GRAP use
the additive form.

References:
----------
- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 5-6.
  https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289
- Bacon, C. R. (2019). *Performance Attribution: History and Progress*.
  CFA Institute Research Foundation. Free PDF:
  https://www.cfainstitute.org/-/media/documents/book/rf-publication/2019/rf-v2019-n4-1-pdf.pdf
- Wikipedia (with citations):
  https://en.wikipedia.org/wiki/Performance_attribution#Geometric_attribution
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._math import MIN_PERIODS, RESIDUAL_TOL
from .._types import LinkedAttribution, PeriodAttribution
from ..errors import AttributionError

if TYPE_CHECKING:
    from collections.abc import Sequence


def link_geometric(periods: Sequence[PeriodAttribution]) -> LinkedAttribution:
    """Link a sequence of single-period attributions geometrically.

    Returns ``interaction = 0`` (folded into selection by construction)
    and ``excess_return = (1 + R_p) / (1 + R_b) − 1``. The multiplicative
    identity ``(1 + allocation) · (1 + selection) − 1 == excess_return``
    holds exactly within floating-point precision.
    """
    if len(periods) < MIN_PERIODS:
        raise AttributionError(
            "link_geometric() requires ≥ 2 periods; for a single period "
            "the multiplicative compounding is a no-op — use the "
            "PeriodAttribution result directly."
        )
    methods = {p.method for p in periods}
    if len(methods) != 1:
        raise AttributionError(
            f"link_geometric() needs a single attribution method per call; "
            f"got {sorted(methods)!r}"
        )

    log_one_plus_alloc = 0.0
    log_one_plus_sel = 0.0
    log_p = 0.0
    log_b = 0.0
    for p in periods:
        if p.benchmark_return <= -1.0 or p.portfolio_return <= -1.0:
            raise AttributionError(
                f"Geometric linking requires period returns > -1 "
                f"(period={p.period!r}, R_p={p.portfolio_return!r}, "
                f"R_b={p.benchmark_return!r})."
            )
        denom_alloc = 1.0 + p.benchmark_return
        a_geom = p.allocation / denom_alloc
        # Selection denominator is the semi-notional 1 + R_s,t = 1 + R_b,t + A_t.
        denom_sel = 1.0 + p.benchmark_return + p.allocation
        if denom_sel <= 0.0:
            raise AttributionError(
                f"Geometric linking: semi-notional return ≤ -100% in "
                f"period {p.period!r} (1 + R_b + A = {denom_sel!r})."
            )
        s_geom = (p.selection + p.interaction) / denom_sel
        if 1.0 + a_geom <= 0.0 or 1.0 + s_geom <= 0.0:
            raise AttributionError(
                f"Geometric linking: per-period geometric effect ≤ -100% "
                f"in period {p.period!r}."
            )
        log_one_plus_alloc += math.log1p(a_geom)
        log_one_plus_sel += math.log1p(s_geom)
        log_p += math.log1p(p.portfolio_return)
        log_b += math.log1p(p.benchmark_return)

    alloc_linked = math.expm1(log_one_plus_alloc)
    sel_linked = math.expm1(log_one_plus_sel)
    r_p_compound = math.expm1(log_p)
    r_b_compound = math.expm1(log_b)
    geom_excess = (1.0 + r_p_compound) / (1.0 + r_b_compound) - 1.0

    multiplicative_residual = (
        (1.0 + alloc_linked) * (1.0 + sel_linked) - 1.0 - geom_excess
    )
    if abs(multiplicative_residual) > RESIDUAL_TOL:  # pragma: no cover
        # Multiplicative identity closure: (1+A)(1+S)-1 == geom_excess
        # holds algebraically once per-period inputs pass the Bacon
        # (2008) sanity gates above. A triggered failure would mean a
        # bug in the log1p/expm1 pipeline — not reachable on valid
        # inputs.
        raise AttributionError(
            f"Geometric linking identity violated: "
            f"(1+A)(1+S)-1 - excess = {multiplicative_residual!r}"
        )

    return LinkedAttribution(
        periods=tuple(periods),
        portfolio_return=r_p_compound,
        benchmark_return=r_b_compound,
        excess_return=geom_excess,
        allocation=alloc_linked,
        selection=sel_linked,
        interaction=0.0,
        method="geometric",
    )
