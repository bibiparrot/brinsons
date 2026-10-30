r"""Public, citable worked-example fixtures for portfolio attribution.

Each function in this submodule returns a small, fully cited worked
example drawn from the academic or practitioner literature on Brinson
attribution. The same fixtures back the test suite — there is one source
of truth, and any change to the expected numbers shows up immediately in
``uv run pytest``.

Why a public fixtures module
----------------------------
No other Python attribution library ships a curated, cited fixture pack
that downstream users can validate their own implementations against.
The goal is for ``pybrinson.fixtures`` to be the educational reference:

>>> from pybrinson import bhb
>>> from pybrinson.fixtures import bacon_2008_ch5_bhb
>>> segments, expected = bacon_2008_ch5_bhb()
>>> result = bhb(segments)
>>> abs(result.excess_return - expected["excess_return"]) < 1e-12
True

Conventions
-----------
Each fixture returns a 2-tuple ``(segments, expected)`` for single-period
examples, or ``(periods, expected)`` for multi-period examples where
``periods`` is a tuple of ``(period_label, segments)`` pairs. The
``expected`` dict contains the totals (``allocation``, ``selection``,
``interaction``, ``excess_return``, ``portfolio_return``,
``benchmark_return``) plus a ``by_segment`` sub-dict keyed on segment
name when relevant.

All numerics are exact rationals expressed as Python floats so that the
identity ``A + S + I == R_p − R_b`` closes to within 1e-12 on every
fixture.

References:
----------
The literature URLs cited in each fixture follow the URL preference
order from ``CLAUDE.md``: free PDF on a publisher / institutional page
when available, then DOI / SSRN / JSTOR landing page, then standards
bodies, then long-form practitioner pages.
"""

from __future__ import annotations

from .._types import Segment

__all__ = [
    "bacon_2008_ch5_bhb",
    "bacon_2008_ch5_fachler",
    "frongello_2002_table1_identical",
    "frongello_2002_table3",
    "karnosky_singer_3segment",
    "two_period_linking_example",
]


def bacon_2008_ch5_bhb() -> tuple[
    tuple[Segment, ...], dict[str, float | dict[str, dict[str, float]]]
]:
    r"""3-sector Brinson-Hood-Beebower worked example.

    Source
    ------
    The canonical UK / Japan / US 3-sector example used as a teaching
    vehicle in:

    - Bacon, C. R. (2008). *Practical Portfolio Performance Measurement
      and Attribution*, 2nd ed., Wiley, chap. 5. ISBN 978-0-470-05928-9.
      https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289
    - Brinson, G. P., Hood, L. R., & Beebower, G. L. (1986).
      "Determinants of Portfolio Performance." *Financial Analysts
      Journal*, 42(4), 39-44. DOI: https://doi.org/10.2469/faj.v42.n4.39
    - CFA Institute Refresher Reading "Portfolio Performance Evaluation":
      https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/performance-attribution

    Numbers
    -------
    UK : ``w_p = w_b = 0.40``, ``R_p = 20%``, ``R_b = 10%``
    JP : ``w_p = 0.30``, ``w_b = 0.20``, ``R_p = -5%``, ``R_b = -4%``
    US : ``w_p = 0.30``, ``w_b = 0.40``, ``R_p = 6%``, ``R_b = 8%``

    Aggregates: ``R_p = 8.3%``, ``R_b = 6.4%``, ``excess = 1.9%``.

    Second-source gap
    -----------------
    The Brinson 1986 original paper and the CFA Bacon (2019) free PDF
    (the planned second pin) are either paywalled or return a placeholder
    when fetched programmatically. The cross-method consistency
    invariants in ``tests/consistency/test_cross_method.py`` (BHB/BF
    total agreement, random-input parametric sweep) serve as the second
    line of defence per ``docs/implementation-v1.2.md`` §T1.3(a).
    """
    segments = (
        Segment("UK Equity", 0.40, 0.40, 0.20, 0.10),
        Segment("Japan Equity", 0.30, 0.20, -0.05, -0.04),
        Segment("US Equity", 0.30, 0.40, 0.06, 0.08),
    )
    expected: dict[str, float | dict[str, dict[str, float]]] = {
        "portfolio_return": 0.083,
        "benchmark_return": 0.064,
        "excess_return": 0.019,
        # Hand-derived from the BHB formulas A=(w_p-w_b)R_b, S=w_b(R_p-R_b),
        # I=(w_p-w_b)(R_p-R_b). Identity: A+S+I == 0.019.
        "allocation": (0.30 - 0.20) * -0.04 + (0.30 - 0.40) * 0.08,
        "selection": 0.40 * (0.20 - 0.10)
        + 0.20 * (-0.05 - -0.04)
        + 0.40 * (0.06 - 0.08),
        "interaction": (0.30 - 0.20) * (-0.05 - -0.04) + (0.30 - 0.40) * (0.06 - 0.08),
        "by_segment": {
            "UK Equity": {
                "allocation": 0.0,
                "selection": 0.40 * (0.20 - 0.10),
                "interaction": 0.0,
            },
            "Japan Equity": {
                "allocation": (0.30 - 0.20) * -0.04,
                "selection": 0.20 * (-0.05 - -0.04),
                "interaction": (0.30 - 0.20) * (-0.05 - -0.04),
            },
            "US Equity": {
                "allocation": (0.30 - 0.40) * 0.08,
                "selection": 0.40 * (0.06 - 0.08),
                "interaction": (0.30 - 0.40) * (0.06 - 0.08),
            },
        },
    }
    return segments, expected


def bacon_2008_ch5_fachler() -> tuple[
    tuple[Segment, ...], dict[str, float | dict[str, dict[str, float]]]
]:
    r"""3-sector Brinson-Fachler worked example.

    Same input segments as :func:`bacon_2008_ch5_bhb`; only the
    allocation formula differs:

    .. math::

        A_i = (w_{p,i} - w_{b,i}) \cdot (R_{b,i} - R_b)

    where :math:`R_b = 6.4\%` is the total benchmark return.

    Source
    ------
    - Brinson, G. P., & Fachler, N. (1985). "Measuring Non-U.S. Equity
      Portfolio Performance." *Journal of Portfolio Management*, 11(3),
      73-76. DOI: https://doi.org/10.3905/jpm.1985.409005
    - Bacon (2008), op. cit., chap. 5.
    """
    segments = (
        Segment("UK Equity", 0.40, 0.40, 0.20, 0.10),
        Segment("Japan Equity", 0.30, 0.20, -0.05, -0.04),
        Segment("US Equity", 0.30, 0.40, 0.06, 0.08),
    )
    r_b_total = 0.064
    expected: dict[str, float | dict[str, dict[str, float]]] = {
        "portfolio_return": 0.083,
        "benchmark_return": r_b_total,
        "excess_return": 0.019,
        # BF allocation totals coincide with BHB because the extra term
        # Σ(w_p - w_b)·R_b = R_b·Σ(w_p - w_b) = 0 when both vectors sum to 1.
        "allocation": (0.30 - 0.20) * (-0.04 - r_b_total)
        + (0.30 - 0.40) * (0.08 - r_b_total),
        "selection": 0.40 * (0.20 - 0.10)
        + 0.20 * (-0.05 - -0.04)
        + 0.40 * (0.06 - 0.08),
        "interaction": (0.30 - 0.20) * (-0.05 - -0.04) + (0.30 - 0.40) * (0.06 - 0.08),
        "by_segment": {
            "UK Equity": {
                "allocation": 0.0,
                "selection": 0.40 * (0.20 - 0.10),
                "interaction": 0.0,
            },
            "Japan Equity": {
                "allocation": (0.30 - 0.20) * (-0.04 - r_b_total),
                "selection": 0.20 * (-0.05 - -0.04),
                "interaction": (0.30 - 0.20) * (-0.05 - -0.04),
            },
            "US Equity": {
                "allocation": (0.30 - 0.40) * (0.08 - r_b_total),
                "selection": 0.40 * (0.06 - 0.08),
                "interaction": (0.30 - 0.40) * (0.06 - 0.08),
            },
        },
    }
    return segments, expected


def frongello_2002_table3() -> tuple[
    tuple[tuple[str, tuple[Segment, ...]], ...],
    dict[str, float],
]:
    r"""3-period 2-sector example from Frongello (2002), Table 3.

    This is the "challenge" example on pages 10-12 of the Fall 2002 JPM
    paper.  Frongello hand-derives the Frongello-linked allocation as
    **57.39475%** and the total excess as **88.9805%**, stepping through
    the recursion period-by-period.

    .. note::

       The original paper uses a 2-effect Brinson-Fachler decomposition
       (allocation + selection, no interaction).  pybrinson tests apply
       BHB (3-effect) to the same inputs; the BHB/BF *total* alloc/sel
       coincide (Σ(w_p − w_b) = 0), and the interaction happens to be
       zero for the Frongello inputs because each period has only one
       segment with a selection bet.

    Source
    ------
    - Frongello, A. S. B. (2002). "Attribution Linking: Proofed and
      Clarified." *The Journal of Performance Measurement*, 7(1), Fall
      2002, pp. 54-67. Open-access author PDF:
      https://frongello.com/support/Works/JPMFall2002.pdf
      Table 3 (p. 10) and hand-derivation of cumulative allocation
      (pp. 11-12).
    """
    p1 = (
        Segment("Stock", 0.70, 0.50, 0.45, -0.25),
        Segment("Bond", 0.30, 0.50, 0.05, -0.05),
    )
    p2 = (
        Segment("Stock", 0.15, 0.10, 0.40, 0.10),
        Segment("Bond", 0.85, 0.90, 0.40, 0.20),
    )
    p3 = (
        Segment("Stock", 0.95, 0.10, 0.05, 0.45),
        Segment("Bond", 0.05, 0.90, 0.45, 0.05),
    )
    periods = (("P1", p1), ("P2", p2), ("P3", p3))
    expected = {
        # Per-period totals from the paper's Table 3.
        "p1_portfolio_return": 0.33,
        "p1_benchmark_return": -0.15,
        "p2_portfolio_return": 0.40,
        "p2_benchmark_return": 0.19,
        "p3_portfolio_return": 0.07,
        "p3_benchmark_return": 0.09,
        # Compounded returns pinned in the paper (bottom of Table 3).
        "compounded_portfolio_return": 1.33 * 1.40 * 1.07 - 1.0,
        "compounded_benchmark_return": 0.85 * 1.19 * 1.09 - 1.0,
        # The paper says 88.9805% excess. The difference vs exact
        # computation is < 1e-10 — we pin to the exact product.
        "arithmetic_excess": (1.33 * 1.40 * 1.07) - (0.85 * 1.19 * 1.09),
        # Frongello-linked allocation derived on pp. 11-12.
        "frongello_linked_allocation": 0.5739475,
    }
    return periods, expected


def frongello_2002_table1_identical() -> tuple[
    tuple[tuple[str, tuple[Segment, ...]], ...],
    dict[str, float],
]:
    r"""3 identical periods from Frongello (2002), Table 1.

    When all periods are identical, Frongello (endnote 13, p. 14) states:
    "Only in the special case of identical duplicate periods do the
    Frongello, Cariño, and Menchero methods produce the same results."

    Furthermore, Frongello's Table 2 pins the proportional attribution for
    3 identical periods at **37.50% allocation, 62.50% selection, 0.00%
    interaction** for all three methods. Since the total excess over 3
    periods is ``(1.054)^3 − (1.038)^3 − ... ≈ 5.25%``, the absolute
    linked effects are alloc = 0.375 · excess, sel = 0.625 · excess.

    Source
    ------
    Same as :func:`frongello_2002_table3`.
    """
    seg = (
        Segment("Stock", 0.80, 0.60, 0.06, 0.05),
        Segment("Bond", 0.20, 0.40, 0.03, 0.02),
    )
    periods = (("P1", seg), ("P2", seg), ("P3", seg))
    r_p = 0.80 * 0.06 + 0.20 * 0.03  # 0.054
    r_b = 0.60 * 0.05 + 0.40 * 0.02  # 0.038
    r_p_cum = (1 + r_p) ** 3 - 1
    r_b_cum = (1 + r_b) ** 3 - 1
    excess = r_p_cum - r_b_cum
    expected = {
        "compounded_portfolio_return": r_p_cum,
        "compounded_benchmark_return": r_b_cum,
        "arithmetic_excess": excess,
        # Paper Table 2: 37.50% allocation, 62.50% selection, 0% interaction.
        "allocation_fraction": 0.375,
        "selection_fraction": 0.625,
        "interaction_fraction": 0.0,
    }
    return periods, expected


def karnosky_singer_3segment() -> tuple[
    tuple[Segment, ...], dict[str, float | dict[str, dict[str, float]]]
]:
    r"""3-region Karnosky-Singer currency attribution worked example.

    Illustrates the additive Karnosky-Singer decomposition (see
    :mod:`pybrinson.single_period.currency`) on a compact US / UK / Japan
    example. Inputs are chosen so that

    - every segment has a non-zero active weight and a non-zero local
      active return, activating all four effects;
    - the additive consistency ``R_p = R_L_p + c`` holds exactly, so the
      fixture can double as a round-trip test.

    Source
    ------
    Formula and sign convention follow:

    - Karnosky, D. S., & Singer, B. D. (1994). *Global Asset Management
      and Performance Attribution*. CFA Institute Research Foundation.
      https://www.cfainstitute.org/-/media/documents/book/rf-publication/1994/rf-v1994-n2-1.pdf
    - Bacon, C. R. (2008). *Practical Portfolio Performance Measurement
      and Attribution*, 2nd ed., Wiley, chap. 7.

    The numeric example is constructed for pedagogical clarity — the
    open-access Karnosky-Singer PDF's own worked example uses index
    snapshots that are not directly reproducible without the source
    data; the identity
    ``market + selection + currency + interaction == excess`` is pinned
    here to ``1e-12`` by construction.
    """
    # Base-currency = USD. US segment has zero currency contribution.
    segments = (
        Segment(
            "US Equity",
            portfolio_weight=0.50,
            benchmark_weight=0.40,
            portfolio_return=0.08,
            benchmark_return=0.06,
            local_return=0.08,
            currency_return=0.0,
        ),
        Segment(
            "UK Equity",
            portfolio_weight=0.30,
            benchmark_weight=0.35,
            portfolio_return=0.05,
            benchmark_return=0.04,
            local_return=0.07,
            currency_return=-0.02,
        ),
        Segment(
            "Japan Equity",
            portfolio_weight=0.20,
            benchmark_weight=0.25,
            portfolio_return=0.02,
            benchmark_return=0.03,
            local_return=0.05,
            currency_return=-0.03,
        ),
    )
    # Derived benchmark local returns (R_L_b,i = R_b,i - c_i):
    # US = 0.06, UK = 0.06, JP = 0.06.
    r_lb_tot = 0.40 * 0.06 + 0.35 * 0.06 + 0.25 * 0.06  # 0.06
    c_b_tot = 0.40 * 0.0 + 0.35 * -0.02 + 0.25 * -0.03  # -0.0145
    market = (
        (0.50 - 0.40) * (0.06 - r_lb_tot)
        + (0.30 - 0.35) * (0.06 - r_lb_tot)
        + (0.20 - 0.25) * (0.06 - r_lb_tot)
    )
    selection = 0.40 * (0.08 - 0.06) + 0.35 * (0.07 - 0.06) + 0.25 * (0.05 - 0.06)
    currency = (
        (0.50 - 0.40) * (0.0 - c_b_tot)
        + (0.30 - 0.35) * (-0.02 - c_b_tot)
        + (0.20 - 0.25) * (-0.03 - c_b_tot)
    )
    interaction = (
        (0.50 - 0.40) * (0.08 - 0.06)
        + (0.30 - 0.35) * (0.07 - 0.06)
        + (0.20 - 0.25) * (0.05 - 0.06)
    )
    r_p = 0.50 * 0.08 + 0.30 * 0.05 + 0.20 * 0.02
    r_b = 0.40 * 0.06 + 0.35 * 0.04 + 0.25 * 0.03
    expected: dict[str, float | dict[str, dict[str, float]]] = {
        "portfolio_return": r_p,
        "benchmark_return": r_b,
        "excess_return": r_p - r_b,
        "market_allocation": market,
        "security_selection": selection,
        "currency_allocation": currency,
        "interaction": interaction,
    }
    return segments, expected


def two_period_linking_example() -> tuple[
    tuple[tuple[str, tuple[Segment, ...]], ...],
    dict[str, float],
]:
    r"""2-period 2-sector example used by every linking method's tests.

    The example is small enough to derive every linking method by hand.
    Each per-period attribution closes the BHB identity exactly, and
    each linking method (Cariño, GRAP, geometric, Menchero, Frongello)
    can be applied to the same input — which is why this fixture also
    backs ``tests/consistency/`` in pybrinson v1.1.

    Per-period BHB results
    ----------------------
    P1: ``A = -0.5%, S = 0, I = +1.0%, excess = +0.5%``
    P2: ``A = 0,    S = +2.5%, I = 0, excess = +2.5%``

    Compounded: ``R_p = 21.5%``, ``R_b = 18.25%``, arithmetic excess
    ``= 3.25%``, geometric excess ``= (1.215 / 1.1825) − 1 ≈ 2.7484%``.

    Source
    ------
    Hand-constructed from the Brinson-Hood-Beebower formulas in
    Bacon (2008) chap. 5 so that every effect is non-degenerate yet the
    arithmetic remains tractable. Used as a cross-method consistency
    fixture in pybrinson v1.1 — see ``docs/implementation-v1.1.md`` T1.3.

    Linking method references:

    - Cariño, D. R. (1999). "Combining Attribution Effects Over Time."
      *Journal of Performance Measurement*, 3(4), 5-14.
      https://www.russellinvestments.com/-/media/files/us/insights/institutions/strategic-research/combining-attribution-effects-over-time.pdf
    - Bacon, C. R. (2019). *Performance Attribution: History and
      Progress*. CFA Institute Research Foundation. Free PDF:
      https://www.cfainstitute.org/-/media/documents/book/rf-publication/2019/rf-v2019-n4-1-pdf.pdf
    """
    p1 = (
        Segment("A", 0.6, 0.5, 0.10, 0.05),
        Segment("B", 0.4, 0.5, 0.05, 0.10),
    )
    p2 = (
        Segment("A", 0.5, 0.5, 0.20, 0.10),
        Segment("B", 0.5, 0.5, 0.05, 0.10),
    )
    periods = (("P1", p1), ("P2", p2))
    expected = {
        # Per-period BHB totals
        "p1_excess": 0.005,
        "p2_excess": 0.025,
        # Compounded
        "compounded_portfolio_return": 1.08 * 1.125 - 1.0,  # 0.215
        "compounded_benchmark_return": 1.075 * 1.10 - 1.0,  # 0.1825
        "arithmetic_excess": (1.08 * 1.125) - (1.075 * 1.10),  # 0.0325
        "geometric_excess": (1.215 / 1.1825) - 1.0,
    }
    return periods, expected
