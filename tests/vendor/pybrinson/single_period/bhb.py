r"""Brinson-Hood-Beebower (BHB) single-period attribution (3-effect).

Formulas
--------
For segment :math:`i` with portfolio weight :math:`w_{p,i}`, benchmark weight
:math:`w_{b,i}`, portfolio return :math:`R_{p,i}` and benchmark return
:math:`R_{b,i}`:

.. math::

   A_i &= (w_{p,i} - w_{b,i}) \cdot R_{b,i}          \quad \text{(allocation)}\\
   S_i &= w_{b,i} \cdot (R_{p,i} - R_{b,i})          \quad \text{(selection)}\\
   I_i &= (w_{p,i} - w_{b,i}) \cdot (R_{p,i} - R_{b,i})  \quad \text{(interaction)}

The identity

.. math::

   \sum_i (A_i + S_i + I_i) = R_p - R_b

holds algebraically whenever both weight vectors sum to 1 and returns are
aggregated as :math:`R = \sum_i w_i R_i`. ``bhb`` raises
:class:`~pybrinson.errors.AttributionError` if the identity ever fails
within ``1e-9`` — never stores a silent residual.

Sign convention
---------------
``A_i > 0`` when the portfolio is overweight (``w_{p,i} > w_{b,i}``) in a
segment whose benchmark return is positive — i.e. a favourable allocation
bet. ``S_i > 0`` when the portfolio's stock-pickers beat the benchmark
inside the segment. ``I_i`` has no direct intuitive sign; it captures the
cross-effect of overweighting a segment *and* outperforming it.

References:
----------
Primary:

- Brinson, G. P., Hood, L. R., & Beebower, G. L. (1986).
  "Determinants of Portfolio Performance."
  *Financial Analysts Journal*, 42(4), 39-44.
  DOI: https://doi.org/10.2469/faj.v42.n4.39

Free CFA Institute reproduction (the 1995 reprint is open-access):

- https://www.cfainstitute.org/-/media/documents/article/premier-pro/determinants-of-portfolio-performance.pdf

Textbook reference with worked examples (used in tests):

- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 5. ISBN 978-0-470-05928-9.
  https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289

Independent cross-checks:

- Wikipedia (with citations): https://en.wikipedia.org/wiki/Performance_attribution
- CFA Institute Refresher Reading "Portfolio Performance Evaluation":
  https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/performance-attribution

Competing implementation referenced as cross-check only:
``fincore/attribution/brinson.py`` lines 82-86 (computes BHB correctly
but stores a silent ``residual`` on identity failure — pybrinson raises
instead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._hierarchy import needs_hierarchy, roll_up

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
from .._math import (
    aggregate_return,
    check_identity,
    reject_empty_segments,
    reject_non_finite,
    validate_weights,
)
from .._types import PeriodAttribution, Segment, SegmentAttribution
from ..errors import AttributionError


def bhb(
    segments: Sequence[Segment],
    *,
    period: str | None = None,
    parents: Mapping[str, str | None] | None = None,
) -> PeriodAttribution:
    """Compute a single-period Brinson-Hood-Beebower attribution.

    Parameters
    ----------
    segments:
        One :class:`~pybrinson.Segment` per classification group. Weights
        must be normalised: ``Σ w_p = Σ w_b = 1`` within ``1e-8``. Returns
        and weights must all be finite (no ``NaN``/``inf``) — pybrinson
        refuses to guess a fill rule.
    period:
        Optional label carried through to the result (e.g. ``"2024-Q1"``).
    parents:
        Optional ``{parent_label: grandparent_label_or_None}`` mapping
        used to walk multi-level hierarchies. ``None`` (default) means
        either flat classification (v1.0 behaviour) or single-level
        nesting via :attr:`Segment.parent`. See
        :mod:`pybrinson._hierarchy` for the roll-up convention.

    Returns:
    -------
    PeriodAttribution
        Totals and per-segment decomposition. The identity
        ``allocation + selection + interaction == excess_return`` holds
        within ``1e-9``; otherwise :class:`AttributionError` is raised.

    Raises:
    ------
    AttributionError
        If ``segments`` is empty; if any weight / return is non-finite;
        if either weight vector does not sum to 1; or if the identity
        fails to close within tolerance (would indicate a numerical
        pathology or a bug).
    """
    if not segments:
        raise AttributionError("bhb() requires at least one Segment")
    reject_empty_segments(segments)

    w_p = [s.portfolio_weight for s in segments]
    w_b = [s.benchmark_weight for s in segments]
    r_p = [s.portfolio_return for s in segments]
    r_b = [s.benchmark_return for s in segments]

    validate_weights(w_p, name="portfolio")
    validate_weights(w_b, name="benchmark")
    reject_non_finite(r_p, name="portfolio returns")
    reject_non_finite(r_b, name="benchmark returns")

    r_p_total = aggregate_return(w_p, r_p)
    r_b_total = aggregate_return(w_b, r_b)
    excess = r_p_total - r_b_total

    by_segment: list[SegmentAttribution] = []
    alloc_total = 0.0
    sel_total = 0.0
    inter_total = 0.0
    for s in segments:
        active = s.portfolio_weight - s.benchmark_weight
        active_ret = s.portfolio_return - s.benchmark_return
        alloc = active * s.benchmark_return
        sel = s.benchmark_weight * active_ret
        inter = active * active_ret
        alloc_total += alloc
        sel_total += sel
        inter_total += inter
        by_segment.append(
            SegmentAttribution(
                name=s.name,
                allocation=alloc,
                selection=sel,
                interaction=inter,
            )
        )

    check_identity(alloc_total, sel_total, inter_total, excess, method="BHB")

    if needs_hierarchy(segments, parents):
        rendered_segments = roll_up(segments, by_segment, parents)
    else:
        rendered_segments = tuple(by_segment)

    return PeriodAttribution(
        period=period,
        portfolio_return=r_p_total,
        benchmark_return=r_b_total,
        excess_return=excess,
        allocation=alloc_total,
        selection=sel_total,
        interaction=inter_total,
        by_segment=rendered_segments,
        method="BHB",
    )
