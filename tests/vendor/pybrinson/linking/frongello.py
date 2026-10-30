r"""Frongello recursive multi-period linking of arithmetic effects.

Method
------
Frongello (2002) proves that the cumulative contribution of an arithmetic
attribution effect (allocation, selection or interaction) over :math:`T`
periods follows the recursion

.. math::

   F_1 &= G_1, \\
   F_t &= G_t \cdot \prod_{s=1}^{t-1}(1 + R_{p,s})
        + R_{b,t} \cdot \sum_{s=1}^{t-1} F_s, \qquad t \ge 2,

where :math:`G_t` is the per-period attribution effect on the channel of
interest, :math:`R_{p,s}` is the **portfolio** total return for period
:math:`s` and :math:`R_{b,t}` is the **benchmark** total return for
period :math:`t`. The linked effect is the sum
:math:`L = \sum_{t=1}^{T} F_t`.

The recursive intuition (page 4 of the paper, "Cumulative Absolute
Attribution and Order Dependence" section) is: each period's
contribution to cumulative outperformance is its single-period attribute
**scaled forward by prior portfolio compounding** plus a correction
**carrying prior cumulative attributes forward at the current
benchmark return**, because the over- or under-performance accumulated
in earlier periods continues to earn the benchmark rate in later
periods.

Closed form
-----------
Unrolling the recursion (a routine telescoping argument, see the proof
sketch below) gives the closed-form scale factor

.. math::

   f_t^{\text{Frongello}}
     = \prod_{s=1}^{t-1}(1 + R_{p,s}) \cdot
       \prod_{s=t+1}^{T}(1 + R_{b,s}),

so that

.. math::

   L = \sum_{t=1}^{T} G_t \cdot f_t^{\text{Frongello}}.

This is **algebraically identical to the GRAP factor**
(see :func:`pybrinson.link_grap`); both methods produce the same
numerical linked effects on every input. The two papers (Frongello 2002
and the GRAP 1997 technical note) reach the same algorithm by different
routes — Frongello via a recursion over base values, GRAP via a
direct closed-form sum-of-products. The Ortec Finance practitioner
review (Leerink & van Breukelen, *Multi-Period Performance Attribution*,
Exhibit 4) explicitly notes that the Cariño, Menchero and Frongello
smoothing methods "provide very similar results to each other"; in the
specific case of Frongello vs GRAP the agreement is exact, not
approximate. pybrinson exposes both names because both appear in the
practitioner literature, and a parametric test
(``tests/linking/test_frongello.py::test_frongello_grap_numerical_equivalence``)
pins the equivalence.

Identity proof (T = 2 sketch)
-----------------------------
With :math:`G_t = R_{p,t} - R_{b,t}` (the per-period excess return):

.. math::

   F_1 + F_2
     &= G_1 + G_2 \cdot (1 + R_{p,1}) + R_{b,2} \cdot G_1 \\
     &= G_1 \cdot (1 + R_{b,2}) + G_2 \cdot (1 + R_{p,1}) \\
     &= (1+R_{p,1})(1+R_{p,2}) - (1+R_{b,1})(1+R_{b,2}) \\
     &= R_p^{\text{cum}} - R_b^{\text{cum}}.

The same telescoping holds for arbitrary :math:`T`; each per-effect
channel inherits the same factor :math:`f_t`, so the additive identity
``A + S + I == compounded R_p − R_b`` is preserved exactly.

Worked example (Frongello 2002, p. 6)
-------------------------------------
Two periods:

* P1: :math:`R_{p,1} = 20\%`, :math:`R_{b,1} = 10\%`,
  allocation :math:`= 5\%`, selection :math:`= 5\%`.
* P2: :math:`R_{p,2} = 10\%`, :math:`R_{b,2} = 5\%`,
  allocation :math:`= 2\%`, selection :math:`= 3\%`.

Frongello-linked allocation :math:`= 5\% + 2\%(1+20\%) + 5\% \cdot 5\%
= 7.65\%`. Selection :math:`= 5\% + 3\%(1+20\%) + 5\% \cdot 5\% = 8.85\%`.
Total :math:`= 16.5\% = 32\% - 15.5\%`. Pinned in
``tests/linking/test_frongello.py``.

Sign / convention choices
-------------------------
Frongello (2002) uses unbarred :math:`R_t` for portfolio return and
:math:`\bar R_t` for benchmark return throughout the proof on pages 4-5.
The recursion as written above keeps that convention: the prefix product
carries **portfolio** returns and the recursive correction term uses the
**current period's benchmark** return on the running sum of prior
contributions. Practitioner write-ups occasionally swap which return
goes prefix vs suffix; that produces a numerically different (and
unpublished) linker. pybrinson follows Frongello's original paper.

Patent / IP note
----------------
Frongello (2002) is published in the Journal of Performance Measurement
and is unencumbered by any patent (unlike Menchero, US 7,249,082 B2,
expired 2024-02-18). The algorithm has been freely reproduced in the R
``PortfolioAttribution`` package, in Bacon (2008) chap. 6, and in the
EDHEC-Risk and Amundi practitioner papers cited below.

References:
----------
Primary:

- Frongello, A. S. B. (2002). "Attribution Linking: Proofed and
  Clarified." *The Journal of Performance Measurement*, 7(1), Fall
  2002, pp. 54-67. Open-access author-hosted PDF:
  https://frongello.com/support/Works/JPMFall2002.pdf
- Frongello, A. (2002). "Linking Single Period Attribution Results."
  *The Journal of Performance Measurement*, 6(3), Spring 2002, pp.
  10-22. Practitioner archive copy:
  https://tsgperformance.com/wp-content/uploads/2017/02/Linking-Single-Period-Attribution-Results.pdf

Cross-checks (free):

- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 6 ("Frongello smoothing" /
  "Multi-period attribution").
  https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289
- Bacon, C. R. (2019). *Performance Attribution: History and Progress*.
  CFA Institute Research Foundation. Free PDF:
  https://www.cfainstitute.org/-/media/documents/book/rf-publication/2019/rf-v2019-n4-1-pdf.pdf
- Leerink, B., & van Breukelen, G. *Multi-Period Performance
  Attribution: Framework for an Allocation Effect Taking Active Weight
  Drift into Account*, Ortec Finance white paper. Exhibit 4 compares
  Cariño / Menchero / Frongello on a real 2013 portfolio and notes the
  near-identity of Cariño and Frongello (and the algebraic equivalence
  of Frongello and the closed-form GRAP factor):
  https://www.ortecfinance.com/-/media/project/ortec/shared/files/whitepapers/ip-multi-period-performance-attribution-ortec-finance_bas-leerink.pdf

History note
------------
Earlier pybrinson releases (≤ 1.1) shipped a transposed implementation
of this function — ``prefix_b · suffix_p`` instead of the published
``prefix_p · suffix_b``. The bug was caught while pinning the v1.2
fixtures against the worked example on page 6 of Frongello (2002): the
old code reported allocation = 7.7% / selection = 8.8% on the paper's
example, where Frongello himself derives 7.65% / 8.85%. The fix
restores the published convention; the additive identity continues to
hold (both forms close it), but the per-effect distribution now matches
the paper.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._math import MIN_PERIODS, check_identity
from .._types import LinkedAttribution, PeriodAttribution
from ..errors import AttributionError

if TYPE_CHECKING:
    from collections.abc import Sequence


def link_frongello(periods: Sequence[PeriodAttribution]) -> LinkedAttribution:
    """Link a sequence of single-period attributions via Frongello recursion.

    Parameters
    ----------
    periods:
        Output of :func:`~pybrinson.bhb` or :func:`~pybrinson.fachler`,
        one per period, in chronological order. Must contain at least
        two periods (use the :class:`~pybrinson.PeriodAttribution`
        directly for a single period) and all use the same single-period
        method.

    Returns:
    -------
    LinkedAttribution
        ``portfolio_return`` and ``benchmark_return`` are the compounded
        single-period returns; ``excess_return = R_p - R_b``; the
        identity ``A + S + I == excess_return`` holds within ``1e-9``.
    """
    if len(periods) < MIN_PERIODS:
        raise AttributionError(
            "link_frongello() requires ≥ 2 periods; for a single period "
            "the recursion collapses to the input — use the "
            "PeriodAttribution result directly."
        )
    methods = {p.method for p in periods}
    if len(methods) != 1:
        raise AttributionError(
            f"link_frongello() needs a single attribution method per call; "
            f"got {sorted(methods)!r}"
        )

    n = len(periods)
    # Prefix-of-portfolio: prefix_p[t] = Π_{s<t}(1 + R_p,s).
    # Per Frongello (2002, pp. 4-5), the per-period attribute is scaled
    # forward by prior portfolio compounding before the benchmark
    # carry-forward correction is added.
    prefix_p = [1.0] * n
    for t in range(1, n):
        prefix_p[t] = prefix_p[t - 1] * (1.0 + periods[t - 1].portfolio_return)

    # Running cumulative linked effect (= Σ_{s≤t} F_s) per channel.
    l_alloc = 0.0
    l_sel = 0.0
    l_inter = 0.0
    for t, p in enumerate(periods):
        # F_t = G_t · prefix_p[t] + R_b,t · (running sum of prior F_s).
        # Read L_X *before* updating it so the carry-forward correction
        # uses Σ_{s<t} F_s, not Σ_{s≤t} F_s.
        f_alloc = p.allocation * prefix_p[t] + p.benchmark_return * l_alloc
        f_sel = p.selection * prefix_p[t] + p.benchmark_return * l_sel
        f_inter = p.interaction * prefix_p[t] + p.benchmark_return * l_inter
        l_alloc += f_alloc
        l_sel += f_sel
        l_inter += f_inter

    log_p = math.fsum(math.log1p(p.portfolio_return) for p in periods)
    log_b = math.fsum(math.log1p(p.benchmark_return) for p in periods)
    r_p_compound = math.expm1(log_p)
    r_b_compound = math.expm1(log_b)
    excess = r_p_compound - r_b_compound

    check_identity(l_alloc, l_sel, l_inter, excess, method="Frongello")

    return LinkedAttribution(
        periods=tuple(periods),
        portfolio_return=r_p_compound,
        benchmark_return=r_b_compound,
        excess_return=excess,
        allocation=l_alloc,
        selection=l_sel,
        interaction=l_inter,
        method="Frongello",
    )
