r"""Brinson-Fachler (BF) single-period attribution (3-effect).

Formulas
--------
With the same notation as :mod:`pybrinson.single_period.bhb`, and writing
:math:`R_b = \sum_i w_{b,i} R_{b,i}` for the total benchmark return:

.. math::

   A_i &= (w_{p,i} - w_{b,i}) \cdot (R_{b,i} - R_b)
       \quad \text{(allocation, BF form)}\\
   S_i &= w_{b,i} \cdot (R_{p,i} - R_{b,i})
       \quad \text{(selection, same as BHB)}\\
   I_i &= (w_{p,i} - w_{b,i}) \cdot (R_{p,i} - R_{b,i})
       \quad \text{(interaction, same as BHB)}

The only difference vs BHB is the allocation term: BF measures each
segment's benchmark return *relative to the total benchmark*, which gives
a more intuitive sign — overweighting a segment that beat the total
benchmark always produces a positive allocation effect, regardless of
the segment's absolute return sign.

The identity :math:`\sum_i (A_i + S_i + I_i) = R_p - R_b` still holds,
because :math:`\sum_i (w_{p,i} - w_{b,i}) \cdot R_b = R_b \cdot
\sum_i (w_{p,i} - w_{b,i}) = 0` whenever both weight vectors sum to 1.

2-effect view
-------------
Many industry reports collapse the 3-effect form to 2 effects by folding
``interaction`` into ``selection``::

    selection_2eff = result.selection + result.interaction

The 2-effect form is what `ppar.attribution` (`ppar/attribution.py:438-451 <https://github.com/JohnDReynolds/portfolio-performance-analytics/blob/3fcf0a91b261c404e11aaac8922835437b698d2e/ppar/attribution.py#L438-L451>`_)
exposes; pybrinson keeps the 3-effect form to avoid losing information,
but a one-line collapse is supported for migration.

Sign convention
---------------
``A_i > 0`` ⇔ overweighting an outperforming segment (or underweighting
an underperforming one). This is the convention in Brinson & Fachler 1985
and the standard CFA / GIPS materials.

References:
----------
Primary:

- Brinson, G. P., & Fachler, N. (1985).
  "Measuring Non-U.S. Equity Portfolio Performance."
  *Journal of Portfolio Management*, 11(3), 73-76.
  DOI: https://doi.org/10.3905/jpm.1985.409005

Free / accessible references:

- CFA Institute Refresher Reading "Portfolio Performance Evaluation":
  https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/performance-attribution
- Wikipedia (with citations): https://en.wikipedia.org/wiki/Performance_attribution#Brinson%E2%80%93Fachler

Textbook reference (worked examples used in tests):

- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 5. ISBN 978-0-470-05928-9.
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


def fachler(
    segments: Sequence[Segment],
    *,
    period: str | None = None,
    parents: Mapping[str, str | None] | None = None,
) -> PeriodAttribution:
    """Compute a single-period Brinson-Fachler attribution (3-effect).

    Parameters
    ----------
    segments:
        See :func:`pybrinson.bhb`.
    period:
        Optional period label.
    parents:
        Optional ``{parent_label: grandparent_label_or_None}`` mapping
        used to walk multi-level hierarchies. See :func:`pybrinson.bhb`
        for the convention.

    Returns:
    -------
    PeriodAttribution
        With ``method == "Brinson-Fachler"``. Identity is enforced.

    Notes:
    -----
    For the 2-effect view used by some commercial reports::

        result = fachler(segments)
        selection_2eff = result.selection + result.interaction
    """
    if not segments:
        raise AttributionError("fachler() requires at least one Segment")
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
        alloc = active * (s.benchmark_return - r_b_total)
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

    check_identity(
        alloc_total, sel_total, inter_total, excess, method="Brinson-Fachler"
    )

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
        method="Brinson-Fachler",
    )
