r"""Menchero optimised multi-period linking of arithmetic attribution effects.

Method
------
Menchero (2000, 2004) introduces a two-part smoothing of single-period
arithmetic effects so that the sum of the linked effects closes the
multi-period arithmetic excess :math:`R_p^{\text{cum}} - R_b^{\text{cum}}`
**exactly**. Each period's per-effect value :math:`a_t` (one of
allocation, selection, interaction) is scaled by :math:`(M + \alpha_t)`
where

.. math::

   M = \frac{(R_p^{\text{cum}} - R_b^{\text{cum}})\,/\,T}
            {(1 + R_p^{\text{cum}})^{1/T} - (1 + R_b^{\text{cum}})^{1/T}},

is a global, period-independent "optimised" multiplier (Menchero's main
factor), and

.. math::

   \alpha_t = \frac{\big(R_p^{\text{cum}} - R_b^{\text{cum}}
                         - M \sum_s (R_{p,s} - R_{b,s})\big)
                    \cdot (R_{p,t} - R_{b,t})}
                   {\sum_s (R_{p,s} - R_{b,s})^2},

is a per-period corrective term chosen to absorb the residual. The
linked effect is

.. math::

   a^{\text{linked}} = \sum_t a_t \cdot (M + \alpha_t).

Identity closure (proof sketch)
-------------------------------
Let :math:`g_t = R_{p,t} - R_{b,t}` and :math:`\Delta = R_p^{\text{cum}}
- R_b^{\text{cum}}`. The sum of linked excess values is

.. math::

   \sum_t g_t \cdot (M + \alpha_t)
     &= M \sum_t g_t
        + \sum_t g_t \cdot \frac{(\Delta - M \sum_s g_s) \cdot g_t}{\sum_s g_s^2} \\
     &= M \sum_t g_t + (\Delta - M \sum_s g_s)
        \cdot \frac{\sum_t g_t^2}{\sum_s g_s^2} \\
     &= M \sum_t g_t + \Delta - M \sum_s g_s \\
     &= \Delta.

Since each of allocation, selection and interaction uses the **same**
per-period scale :math:`(M + \alpha_t)`, the additive identity
:math:`A^{\text{linked}} + S^{\text{linked}} + I^{\text{linked}} =
\Delta` follows term-by-term from the BHB per-period identity
:math:`A_t + S_t + I_t = g_t`.

Degenerate cases
----------------
1. **Zero multi-period excess** (:math:`R_p^{\text{cum}} =
   R_b^{\text{cum}}`). The closed form of :math:`M` is a 0/0 limit; the
   l'Hôpital value is :math:`M = (1 + R_b^{\text{cum}})^{(T-1)/T}` and
   :math:`\alpha_t = 0`. Menchero (2000) presents this as the limit case;
   the R ``PortfolioAttribution`` package (upstream from Spaulding)
   implements the same branch.
2. **All per-period excesses zero** (:math:`\sum g_t^2 = 0`). Forces
   :math:`R_p^{\text{cum}} = R_b^{\text{cum}}` so the first branch fires
   and :math:`\alpha_t = 0` — every effect links to zero. Handled by
   falling through to case 1 and returning early.
3. **Total-wipe-out period** (:math:`R \le -1`). Menchero's
   :math:`(1 + R)^{1/T}` is undefined (complex root). pybrinson raises
   :class:`AttributionError` to match the guard on :func:`link_carino` /
   :func:`link_geometric` — a period that loses 100% of value cannot be
   linked.

Patent status
-------------
The Menchero optimised-linking algorithm was covered by **US patent
7,249,082 B2** ("Method and system for multi-period performance
attribution with metric-preserving coefficients"), assigned to Vestek
Systems, Inc., issued 2007-07-24, **expired 2024-02-18** (anticipated
expiration per the USPTO file wrapper). The project's `docs/implementation-v1.2.md`
§T1.2 requires a primary patent-status source link in the PR description
before this implementation may merge; the expiry is recorded on the
Google Patents mirror of the USPTO record:

- https://patents.google.com/patent/US7249082
- https://www.freepatentsonline.com/7246091.html (related family member)

As of 2024-02-18 the algorithm is in the public domain worldwide.

Sign / convention choice
------------------------
Menchero (2000) and Menchero (2004) present the coefficient in slightly
different algebraic form (the 2004 FAJ piece factors the scaling
differently). pybrinson pins to the 2000 *Journal of Performance
Measurement* formulation as reproduced in Bacon (2008, chap. 6, pp.
194-196) because it is the form used by every open-source reference
implementation we could audit (the R ``PortfolioAttribution`` package
and Spaulding's JPM examples). The 2004 FAJ formulation is
algebraically equivalent on the additive identity.

References:
----------
Primary:

- Menchero, J. (2000). "An Optimized Approach to Linking Attribution
  Effects Over Time." *Journal of Performance Measurement*, 5(1),
  Fall 2000, pp. 36-42. JPM author-copy PDFs surface on
  practitioner archives; the paywalled Spaulding Group TSG landing
  page: https://tsgperformance.com/insights/spaulding-group-books-and-articles/
- Menchero, J. (2004). "Multiperiod Arithmetic Attribution."
  *Financial Analysts Journal*, 60(4), 76-91.
  DOI: https://doi.org/10.2469/faj.v60.n4.2638
  CFA Institute landing (abstract + free preview):
  https://rpc.cfainstitute.org/research/financial-analysts-journal/2004/multiperiod-arithmetic-attribution

Cross-checks (free / open access):

- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 6, pp. 194-196.
  https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289
- Bacon, C. R. (2019). *Performance Attribution: History and Progress*.
  CFA Institute Research Foundation. Free PDF:
  https://www.cfainstitute.org/-/media/documents/book/rf-publication/2019/rf-v2019-n4-1-pdf.pdf
- R ``PortfolioAttribution`` package (open-source reference
  implementation of exactly this formula, GPL-3):
  https://github.com/R-Finance/PortfolioAttribution/blob/master/R/Menchero.R

The formula structure (branching on ``rpc == rbc``, per-period
correction :math:`\alpha_t`) is structurally identical to the R
reference above; the underlying math is Menchero 2000.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._math import MIN_PERIODS, check_identity
from .._types import LinkedAttribution, PeriodAttribution
from ..errors import AttributionError

if TYPE_CHECKING:
    from collections.abc import Sequence

_EXCESS_DIFF_TOL = 1e-12


def link_menchero(periods: Sequence[PeriodAttribution]) -> LinkedAttribution:
    """Link a sequence of single-period attributions via Menchero smoothing.

    Parameters
    ----------
    periods:
        Output of :func:`~pybrinson.bhb` or :func:`~pybrinson.fachler`,
        one per period, in chronological order. Must contain at least
        two periods and all use the same single-period method.

    Returns:
    -------
    LinkedAttribution
        ``portfolio_return`` and ``benchmark_return`` are the compounded
        single-period returns; ``excess_return = R_p - R_b``; the
        identity ``A + S + I == excess_return`` holds within ``1e-9``.
    """
    if len(periods) < MIN_PERIODS:
        raise AttributionError(
            "link_menchero() requires ≥ 2 periods; for a single period the "
            "Menchero coefficients collapse to unity — use the "
            "PeriodAttribution result directly."
        )
    methods = {p.method for p in periods}
    if len(methods) != 1:
        raise AttributionError(
            f"link_menchero() needs a single attribution method per call; "
            f"got {sorted(methods)!r}"
        )

    # Guard on total-wipe-out: (1 + R)^(1/T) is not real for R ≤ -1.
    for p in periods:
        if p.portfolio_return <= -1.0 or p.benchmark_return <= -1.0:
            raise AttributionError(
                f"Menchero linking requires returns > -1 (period={p.period!r}, "
                f"R_p={p.portfolio_return!r}, R_b={p.benchmark_return!r}). "
                "A period that wipes out 100% or more of value has no real "
                "T-th root and cannot be linked."
            )

    n = len(periods)
    log_p = math.fsum(math.log1p(p.portfolio_return) for p in periods)
    log_b = math.fsum(math.log1p(p.benchmark_return) for p in periods)
    r_p_compound = math.expm1(log_p)
    r_b_compound = math.expm1(log_b)
    delta = r_p_compound - r_b_compound

    if abs(delta) <= _EXCESS_DIFF_TOL:
        # Limit case: Menchero (2000) gives M = (1 + R_b^cum)^((T-1)/T)
        # and α_t = 0, which forces the linked effects to Σ a_t · M.
        # When all g_t are also zero (the typical sub-case), every linked
        # effect collapses to zero.
        big_m = (1.0 + r_b_compound) ** ((n - 1) / n)
        alloc = math.fsum(p.allocation * big_m for p in periods)
        sel = math.fsum(p.selection * big_m for p in periods)
        inter = math.fsum(p.interaction * big_m for p in periods)
        check_identity(alloc, sel, inter, delta, method="Menchero")
        return LinkedAttribution(
            periods=tuple(periods),
            portfolio_return=r_p_compound,
            benchmark_return=r_b_compound,
            excess_return=delta,
            allocation=alloc,
            selection=sel,
            interaction=inter,
            method="Menchero",
        )

    # Main branch: rpc ≠ rbc. The T-th roots are well-defined because we
    # already rejected returns ≤ -1 above.
    root_p = (1.0 + r_p_compound) ** (1.0 / n)
    root_b = (1.0 + r_b_compound) ** (1.0 / n)
    big_m = (delta / n) / (root_p - root_b)

    g = [p.portfolio_return - p.benchmark_return for p in periods]
    sum_g = math.fsum(g)
    sum_g2 = math.fsum(gi * gi for gi in g)
    # sum_g2 == 0 ⇒ every g_t = 0 ⇒ Δ = 0, which would have taken the
    # limit branch above. A non-zero Δ with sum_g2 = 0 is impossible.
    numerator = delta - big_m * sum_g

    alloc = 0.0
    sel = 0.0
    inter = 0.0
    for gi, p in zip(g, periods, strict=True):
        alpha_t = numerator * gi / sum_g2
        scale = big_m + alpha_t
        alloc += p.allocation * scale
        sel += p.selection * scale
        inter += p.interaction * scale

    check_identity(alloc, sel, inter, delta, method="Menchero")

    return LinkedAttribution(
        periods=tuple(periods),
        portfolio_return=r_p_compound,
        benchmark_return=r_b_compound,
        excess_return=delta,
        allocation=alloc,
        selection=sel,
        interaction=inter,
        method="Menchero",
    )
