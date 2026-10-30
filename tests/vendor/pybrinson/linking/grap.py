r"""GRAP multi-period linking of arithmetic attribution effects.

Method
------
The GRAP (Groupe de Recherche en Attribution de Performance) factor for
period :math:`t` in a sequence of :math:`T` periods is

.. math::

   f_t = \prod_{s=1}^{t-1}(1 + R_{p,s}) \cdot \prod_{s=t+1}^{T}(1 + R_{b,s}),

i.e. compound past portfolio returns and future benchmark returns up to
period :math:`t`. The linked allocation effect is

.. math::

   A^{\text{linked}} = \sum_{t=1}^{T} A_t \cdot f_t,

and analogously for selection and interaction.

Identity proof (sketch)
-----------------------
Define :math:`E_t = R_{p,t} - R_{b,t}`. Then

.. math::

   \sum_t E_t \cdot f_t
     = \sum_t \Big(\prod_{s<t}(1 + R_{p,s})\Big) \cdot
       (R_{p,t} - R_{b,t}) \cdot
       \Big(\prod_{s>t}(1 + R_{b,s})\Big),

which telescopes to :math:`\prod_t (1 + R_{p,t}) - \prod_t (1 + R_{b,t})
= R_p - R_b` (the compounded arithmetic excess). Each per-effect series
inherits the same factor :math:`f_t`, so the additive identity is
preserved exactly.

References:
----------
Primary:

- Groupe de Recherche en Attribution de Performance (1997).
  *Synthèse des modèles d'attribution de performance*. Paris: GRAP /
  AFG-ASFFI. (French industry standard, the original technical note is
  not freely hosted; pybrinson works from the formulation reproduced in
  Bacon and in the EDHEC and Amundi practitioner papers below.)

Reproductions and cross-checks (free):

- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 6 (the GRAP linking method is
  presented side-by-side with Cariño and Frongello).
  https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289
- EDHEC-Risk Institute, "Multi-period attribution: a survey" (free):
  https://risk.edhec.edu/sites/risk/files/edhec-publication-multi-period-portfolio-performance-attribution.pdf
- Amundi practitioner paper, "Multi-period attribution":
  https://research-center.amundi.com/article/multi-period-performance-attribution
- Wikipedia (with citations):
  https://en.wikipedia.org/wiki/Performance_attribution#Linking_attribution_effects
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._math import MIN_PERIODS, check_identity
from .._types import LinkedAttribution, PeriodAttribution
from ..errors import AttributionError

if TYPE_CHECKING:
    from collections.abc import Sequence


def link_grap(periods: Sequence[PeriodAttribution]) -> LinkedAttribution:
    """Link a sequence of single-period attributions via GRAP factors.

    See :func:`pybrinson.link_carino` for parameter conventions; both
    methods produce additive linked effects that close the identity
    ``A + S + I == compounded R_p − compounded R_b`` within ``1e-9``.
    """
    if len(periods) < MIN_PERIODS:
        raise AttributionError(
            "link_grap() requires ≥ 2 periods; for a single period the "
            "GRAP factor is identically 1 — use the PeriodAttribution "
            "result directly."
        )
    methods = {p.method for p in periods}
    if len(methods) != 1:
        raise AttributionError(
            f"link_grap() needs a single attribution method per call; "
            f"got {sorted(methods)!r}"
        )

    n = len(periods)
    # Cumulative compounded portfolio returns up to (but excluding) period t.
    prefix_p = [1.0] * n
    for t in range(1, n):
        prefix_p[t] = prefix_p[t - 1] * (1.0 + periods[t - 1].portfolio_return)
    # Cumulative compounded benchmark returns from (but excluding) period t.
    suffix_b = [1.0] * n
    for t in range(n - 2, -1, -1):
        suffix_b[t] = suffix_b[t + 1] * (1.0 + periods[t + 1].benchmark_return)

    alloc = 0.0
    sel = 0.0
    inter = 0.0
    for t, p in enumerate(periods):
        f_t = prefix_p[t] * suffix_b[t]
        alloc += p.allocation * f_t
        sel += p.selection * f_t
        inter += p.interaction * f_t

    log_p = math.fsum(math.log1p(p.portfolio_return) for p in periods)
    log_b = math.fsum(math.log1p(p.benchmark_return) for p in periods)
    r_p_compound = math.expm1(log_p)
    r_b_compound = math.expm1(log_b)
    excess = r_p_compound - r_b_compound

    check_identity(alloc, sel, inter, excess, method="GRAP")

    return LinkedAttribution(
        periods=tuple(periods),
        portfolio_return=r_p_compound,
        benchmark_return=r_b_compound,
        excess_return=excess,
        allocation=alloc,
        selection=sel,
        interaction=inter,
        method="GRAP",
    )
