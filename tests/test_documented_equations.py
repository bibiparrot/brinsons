"""Independent scalar checks for the equations in the Sphinx method guide."""
from math import expm1, log1p

import polars as pl
import pytest

from brinsons import (
    bhb, fachler, link_carino, link_grap, link_frongello,
    link_menchero, link_geometric, currency_attribution, currency_premiums,
)


def facts():
    return pl.DataFrame({
        "period": [1, 1, 2, 2],
        "segment": ["Equity", "Bonds", "Equity", "Bonds"],
        "portfolio_weight": [.6, .4, .55, .45],
        "benchmark_weight": [.5, .5, .5, .5],
        "portfolio_return": [.12, .04, .03, .01],
        "benchmark_return": [.10, .06, .02, .015],
    })


@pytest.mark.parametrize("method", [bhb, fachler])
@pytest.mark.parametrize("effects", [2, 3])
def test_documented_single_period_rows(method, effects):
    result = method(facts(), effects=effects)
    for period in [1, 2]:
        rows = facts().filter(pl.col("period") == period).to_dicts()
        benchmark = sum(x["benchmark_weight"] * x["benchmark_return"] for x in rows)
        portfolio = sum(x["portfolio_weight"] * x["portfolio_return"] for x in rows)
        for row in rows:
            dw = row["portfolio_weight"] - row["benchmark_weight"]
            dr = row["portfolio_return"] - row["benchmark_return"]
            reference = benchmark if method is fachler else 0.0
            got = result.detail.filter(
                (pl.col("period") == period) & (pl.col("segment") == row["segment"])
            ).row(0, named=True)
            assert got["allocation"] == pytest.approx(dw * (row["benchmark_return"] - reference))
            assert got["selection"] == pytest.approx(
                (row["benchmark_weight"] if effects == 3 else row["portfolio_weight"]) * dr
            )
            assert got["interaction"] == pytest.approx(dw * dr if effects == 3 else 0.0)
        total = result.summary.filter(pl.col("period") == period).row(0, named=True)
        assert sum(total[x] for x in ("allocation", "selection", "interaction")) == pytest.approx(
            portfolio - benchmark
        )


def _factors(method, periods):
    p = [row["portfolio_return"] for row in periods]
    b = [row["benchmark_return"] for row in periods]
    g = [a - z for a, z in zip(p, b)]
    P = (1 + p[0]) * (1 + p[1]) - 1
    B = (1 + b[0]) * (1 + b[1]) - 1
    E = P - B
    if method == "carino":
        def k(a, z):
            return (log1p(a) - log1p(z)) / (a - z) if abs(a - z) >= 1e-12 else 1 / (1 + z)
        return [k(a, z) / k(P, B) for a, z in zip(p, b)]
    if method in ("grap", "frongello"):
        return [1 + b[1], 1 + p[0]]
    M = ((E / 2) / ((1 + P) ** .5 - (1 + B) ** .5)
         if abs(E) >= 1e-12 else (1 + B) ** .5)
    residual = E - M * sum(g)
    return [M + residual * x / sum(y * y for y in g) for x in g]


@pytest.mark.parametrize("name,method", [
    ("carino", link_carino), ("grap", link_grap),
    ("frongello", link_frongello), ("menchero", link_menchero),
])
def test_documented_arithmetic_link_factors(name, method):
    original = fachler(facts())
    periods = original.summary.sort("period").to_dicts()
    factors = _factors(name, periods)
    linked = method(original)
    for period, factor in zip([1, 2], factors):
        detail = linked.detail.filter(pl.col("period") == period)
        assert detail["factor"].to_list() == pytest.approx([factor] * detail.height)
        for effect in ("allocation", "selection", "interaction"):
            for row in detail.to_dicts():
                assert row["linked_" + effect] == pytest.approx(row[effect] * factor)
    p = (1 + periods[0]["portfolio_return"]) * (1 + periods[1]["portfolio_return"]) - 1
    b = (1 + periods[0]["benchmark_return"]) * (1 + periods[1]["benchmark_return"]) - 1
    assert linked.summary["excess_return"][0] == pytest.approx(p - b)
    assert sum(linked.summary[x][0] for x in ("allocation", "selection", "interaction")) == pytest.approx(p - b)


def test_documented_geometric_product():
    periods = fachler(facts()).summary.sort("period").to_dicts()
    allocation = 1.0
    selection = 1.0
    for row in periods:
        allocation *= 1 + row["allocation"] / (1 + row["benchmark_return"])
        selection *= 1 + (row["selection"] + row["interaction"]) / (
            1 + row["benchmark_return"] + row["allocation"]
        )
    got = link_geometric(fachler(facts())).summary.row(0, named=True)
    assert got["allocation"] == pytest.approx(allocation - 1)
    assert got["selection"] == pytest.approx(selection - 1)
    assert got["interaction"] == 0
    assert got["excess_return"] == pytest.approx(allocation * selection - 1)


def test_documented_additive_currency_rows():
    data = facts().filter(pl.col("period") == 1).with_columns(
        pl.Series("currency_return", [.01, -.005]),
        pl.Series("local_return", [.11, .045]),
    )
    got = {row["segment"]: row for row in currency_attribution(data).detail.to_dicts()}
    rows = data.to_dicts()
    local_b = [x["benchmark_return"] - x["currency_return"] for x in rows]
    mean_local = sum(x["benchmark_weight"] * y for x, y in zip(rows, local_b))
    mean_fx = sum(x["benchmark_weight"] * x["currency_return"] for x in rows)
    for row, lb in zip(rows, local_b):
        out = got[row["segment"]]
        dw = row["portfolio_weight"] - row["benchmark_weight"]
        dr = row["local_return"] - lb
        assert out["market_allocation"] == pytest.approx(dw * (lb - mean_local))
        assert out["security_selection"] == pytest.approx(row["benchmark_weight"] * dr)
        assert out["currency_allocation"] == pytest.approx(dw * (row["currency_return"] - mean_fx))
        assert out["interaction"] == pytest.approx(dw * dr)


def test_documented_currency_log_premiums():
    markets = pl.DataFrame({
        "period": [1, 1], "segment": ["A", "B"],
        "portfolio_weight": [.6, .4], "benchmark_weight": [.5, .5],
        "portfolio_return": [expm1(.08), expm1(.02)],
        "benchmark_return": [expm1(.05), expm1(.03)],
        "local_cash_return": [0., 0.],
    })
    currencies = pl.DataFrame({
        "period": [1, 1], "segment": ["EUR", "USD"],
        "portfolio_weight": [.7, .3], "benchmark_weight": [.5, .5],
        "portfolio_return": [expm1(.04), expm1(.015)],
        "benchmark_return": [expm1(.03), expm1(.01)],
    })
    market, fx, summary = currency_premiums(markets, currencies)
    assert market["market_allocation"].to_list() == pytest.approx([.001, .001])
    assert market["security_selection"].to_list() == pytest.approx([.018, -.004])
    assert fx["currency_allocation"].to_list() == pytest.approx([.002, .002])
    assert fx["hedge_selection"].to_list() == pytest.approx([.007, .0015])
    total = summary.row(0, named=True)
    assert total["active_log_return"] == pytest.approx(.0285)
