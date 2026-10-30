r"""Karnosky-Singer single-period currency attribution.

Decomposes the excess return of a global (multi-currency) portfolio into
four effects per segment — local-market allocation, security selection,
currency allocation, and interaction — such that the sum across segments
reconstitutes the base-currency arithmetic excess ``R_p − R_b``.

Formulas
--------
For segment :math:`i` with portfolio / benchmark weights :math:`w_{p,i},
w_{b,i}`, base-currency returns :math:`R_{p,i}, R_{b,i}`, local-currency
portfolio return :math:`R^L_{p,i}`, and currency contribution
:math:`c_i` (shared by portfolio and benchmark — both hold the segment
in the same local currency), the benchmark local return is derived as
:math:`R^L_{b,i} \equiv R_{b,i} - c_i` under the additive Karnosky-Singer
convention (continuously compounded returns). Define benchmark local and
currency totals:

.. math::

   R^L_{b,\text{tot}} &= \sum_i w_{b,i} \, R^L_{b,i} \\
   c_{b,\text{tot}}   &= \sum_i w_{b,i} \, c_i

The four per-segment effects are then

.. math::

   M_i &= (w_{p,i} - w_{b,i})(R^L_{b,i} - R^L_{b,\text{tot}})
          \quad \text{(market allocation — BF form)} \\
   S_i &= w_{b,i}(R^L_{p,i} - R^L_{b,i})
          \quad \text{(security selection)} \\
   C_i &= (w_{p,i} - w_{b,i})(c_i - c_{b,\text{tot}})
          \quad \text{(currency allocation — BF form)} \\
   I_i &= (w_{p,i} - w_{b,i})(R^L_{p,i} - R^L_{b,i})
          \quad \text{(interaction)}

Summing over :math:`i` and using :math:`\sum w_{p,i} = \sum w_{b,i} = 1`
plus :math:`R_{p,i} = R^L_{p,i} + c_i`, the four terms collapse to
:math:`R_p - R_b`. pybrinson enforces this identity within ``1e-9`` via
:func:`pybrinson._math.check_identity`.

Sign / convention choices
-------------------------
pybrinson pins the **additive (continuously compounded) Karnosky-Singer
convention**: ``portfolio_return = local_return + currency_return``.
Ankrim & Hensel (1994) use a multiplicative form
``(1 + R^L)(1 + c) - 1`` and an explicit forward-premium leg. We do NOT
implement a forward-premium / hedged-benchmark leg in v1.3: callers who
work under a hedged benchmark should feed currency-hedged base-currency
returns and set ``currency_return`` to the residual unhedged
contribution. This matches Bacon (2008) chap. 7's "naive" currency model
and keeps the decomposition identity-closing without needing forward
rates as an extra input.

References:
----------
Primary:

- Karnosky, D. S., & Singer, B. D. (1994). *Global Asset Management and
  Performance Attribution*. CFA Institute Research Foundation
  Publications, 1994(2).
  https://www.cfainstitute.org/-/media/documents/book/rf-publication/1994/rf-v1994-n2-1.pdf
- Ankrim, E. M., & Hensel, C. R. (1994). "Multicurrency Performance
  Attribution." *Financial Analysts Journal*, 50(2), 29-35.
  DOI: https://doi.org/10.2469/faj.v50.n2.29

Textbook cross-checks:

- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 7 "Multi-currency attribution".
  https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289
- Allen, G. C. (1991). "Performance Attribution for Global Equity
  Portfolios." *Journal of Portfolio Management*, 18(1), 59-65.
  DOI: https://doi.org/10.3905/jpm.1991.409385

Independent long-form explainer:

- https://en.wikipedia.org/wiki/Performance_attribution#Currency_attribution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._math import (
    RESIDUAL_TOL,
    aggregate_return,
    check_identity,
    reject_empty_segments,
    reject_non_finite,
    validate_weights,
)
from .._types import CurrencyPeriodAttribution, CurrencySegmentAttribution
from ..errors import AttributionError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._types import Segment


def _validate_currency_inputs(
    segments: Sequence[Segment],
) -> tuple[list[float], list[float]]:
    """Check required fields and additive consistency; return (R_L_p, c)."""
    for s in segments:
        if s.local_return is None or s.currency_return is None:
            raise AttributionError(
                f"segment {s.name!r} is missing local_return and/or "
                "currency_return. Karnosky-Singer currency attribution "
                "requires both fields on every segment."
            )
    r_lp: list[float] = [s.local_return for s in segments]  # type: ignore[misc]
    c: list[float] = [s.currency_return for s in segments]  # type: ignore[misc]
    reject_non_finite(r_lp, name="local returns")
    reject_non_finite(c, name="currency returns")
    for s, lp, ci in zip(segments, r_lp, c, strict=True):
        if abs(s.portfolio_return - (lp + ci)) > RESIDUAL_TOL:
            raise AttributionError(
                f"segment {s.name!r} violates additive consistency: "
                f"portfolio_return={s.portfolio_return!r} but "
                f"local_return + currency_return = {lp + ci!r}. "
                "pybrinson pins the additive Karnosky-Singer convention — "
                "see src/pybrinson/single_period/currency.py for details."
            )
    return r_lp, c


def currency_attribution(
    segments: Sequence[Segment],
    *,
    period: str | None = None,
) -> CurrencyPeriodAttribution:
    """Compute a single-period Karnosky-Singer currency attribution.

    Each :class:`~pybrinson.Segment` must supply both ``local_return``
    (portfolio's local-currency segment return) and ``currency_return``
    (currency contribution shared by portfolio and benchmark for the
    segment). The benchmark local return is derived as
    ``benchmark_return - currency_return`` under the additive convention
    documented at the top of this module.

    Parameters
    ----------
    segments:
        One Segment per classification group. Weights must sum to 1
        within ``1e-8``; every return must be finite; every segment must
        carry both ``local_return`` and ``currency_return``. The additive
        consistency ``portfolio_return == local_return + currency_return``
        is checked within ``1e-9``.
    period:
        Optional label carried through to the result.

    Returns:
    -------
    CurrencyPeriodAttribution
        Totals and per-segment 4-effect decomposition. The identity
        ``market_allocation + security_selection + currency_allocation
        + interaction == excess_return`` holds within ``1e-9``.

    Raises:
    ------
    AttributionError
        On empty input, missing currency fields, non-finite values,
        non-normalised weights, additive-consistency violation, or
        identity failure.
    """
    if not segments:
        raise AttributionError("currency_attribution() requires at least one Segment")
    reject_empty_segments(segments)

    w_p = [s.portfolio_weight for s in segments]
    w_b = [s.benchmark_weight for s in segments]
    r_p = [s.portfolio_return for s in segments]
    r_b = [s.benchmark_return for s in segments]
    validate_weights(w_p, name="portfolio")
    validate_weights(w_b, name="benchmark")
    reject_non_finite(r_p, name="portfolio returns")
    reject_non_finite(r_b, name="benchmark returns")

    r_lp_f, c_f = _validate_currency_inputs(segments)

    # Benchmark local return derived from additive convention.
    r_lb = [rb - ci for rb, ci in zip(r_b, c_f, strict=True)]

    r_p_total = aggregate_return(w_p, r_p)
    r_b_total = aggregate_return(w_b, r_b)
    r_lb_total = aggregate_return(w_b, r_lb)
    c_b_total = aggregate_return(w_b, c_f)
    excess = r_p_total - r_b_total

    by_segment: list[CurrencySegmentAttribution] = []
    m_total = 0.0
    s_total = 0.0
    cc_total = 0.0
    i_total = 0.0
    for s, lp, ci, lb in zip(segments, r_lp_f, c_f, r_lb, strict=True):
        active = s.portfolio_weight - s.benchmark_weight
        active_local = lp - lb
        m = active * (lb - r_lb_total)
        sel = s.benchmark_weight * active_local
        cc = active * (ci - c_b_total)
        inter = active * active_local
        m_total += m
        s_total += sel
        cc_total += cc
        i_total += inter
        by_segment.append(
            CurrencySegmentAttribution(
                name=s.name,
                market_allocation=m,
                security_selection=sel,
                currency_allocation=cc,
                interaction=inter,
            )
        )

    # Reuse the 3-effect identity checker by folding currency_allocation
    # into the "selection" slot (any stable bucketing works — the check
    # is just a sum).
    check_identity(
        m_total + cc_total,
        s_total,
        i_total,
        excess,
        method="Karnosky-Singer",
    )

    return CurrencyPeriodAttribution(
        period=period,
        portfolio_return=r_p_total,
        benchmark_return=r_b_total,
        excess_return=excess,
        market_allocation=m_total,
        security_selection=s_total,
        currency_allocation=cc_total,
        interaction=i_total,
        by_segment=tuple(by_segment),
        method="Karnosky-Singer",
    )
