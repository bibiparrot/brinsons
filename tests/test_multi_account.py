"""Multi-account identities checked independently of the Polars expressions."""
from collections import defaultdict
from pathlib import Path

import polars as pl
import pytest

from brinsons import (
    AttributionError, bhb, macro_micro_attribution,
    value_weighted_total_attribution, hierarchical_decision_attribution,
    link_macro_micro_attribution, link_hierarchical_decision_attribution,
    link_carino, link_grap, link_frongello, link_menchero, link_geometric,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "multi_account" / "positions.csv"


def positions():
    return pl.read_csv(DATA)


def _sum(rows, key, value):
    return sum(r[key] * r[value] for r in rows)


def _independent_notionals(rows):
    """Pure Python implementation of the fixed decision order."""
    pv = sum(r["portfolio_value"] for r in rows)
    bv = sum(r["benchmark_value"] for r in rows)
    accounts = defaultdict(list)
    sectors = defaultdict(list)
    for row in rows:
        accounts[row["account"]].append(row)
        sectors[(row["account"], row["sector"])].append(row)
    n0 = _sum(rows, "benchmark_value", "benchmark_return") / bv
    n4 = _sum(rows, "portfolio_value", "portfolio_return") / pv
    n1 = n2 = n3 = 0.0
    for account, account_rows in accounts.items():
        pa = sum(r["portfolio_value"] for r in account_rows)
        ba = sum(r["benchmark_value"] for r in account_rows)
        account_benchmark_return = _sum(account_rows, "benchmark_value", "benchmark_return") / ba
        n1 += pa / pv * account_benchmark_return
        for (owner, _), sector_rows in sectors.items():
            if owner != account:
                continue
            ps = sum(r["portfolio_value"] for r in sector_rows)
            bs = sum(r["benchmark_value"] for r in sector_rows)
            sector_benchmark_return = _sum(sector_rows, "benchmark_value", "benchmark_return") / bs
            n2 += pa / pv * ps / pa * sector_benchmark_return
            for row in sector_rows:
                n3 += pa / pv * ps / pa * row["portfolio_value"] / ps * row["benchmark_return"]
    return [n0, n1, n2, n3, n4]


def test_three_methods_reconcile_to_direct_values():
    data = positions()
    macro = macro_micro_attribution(data)
    pooled = value_weighted_total_attribution(data)
    decision = hierarchical_decision_attribution(data)
    assert data.height == 36
    for period in data["period"].unique().sort().to_list():
        rows = data.filter(pl.col("period") == period).to_dicts()
        expected = _independent_notionals(rows)
        a = macro.integrated_summary.filter(pl.col("period") == period).row(0, named=True)
        b = pooled.summary.filter(pl.col("period") == period).row(0, named=True)
        c = decision.summary.filter(pl.col("period") == period).row(0, named=True)
        assert a["portfolio_return"] == pytest.approx(expected[-1])
        assert a["benchmark_return"] == pytest.approx(expected[0])
        for result in (a, b, c):
            assert result["excess_return"] == pytest.approx(expected[-1] - expected[0])
        assert a["total_effect"] == pytest.approx(a["excess_return"])
        assert c["total_effect"] == pytest.approx(c["excess_return"])
        notionals = decision.notionals.filter(pl.col("period") == period).sort("stage_index")
        assert notionals["return"].to_list() == pytest.approx(expected)
        for index, effect in enumerate(("account_allocation", "sector_allocation",
                                        "security_allocation", "security_selection")):
            assert c[effect] == pytest.approx(expected[index + 1] - expected[index])


def test_macro_micro_replaces_account_selection_without_double_counting():
    result = macro_micro_attribution(positions())
    for row in result.integrated_detail.to_dicts():
        micro = sum(row["weighted_micro_" + name] for name in
                    ("allocation", "selection", "interaction"))
        assert micro == pytest.approx(row["macro_selection"] + row["macro_interaction"])
        assert row["total_effect"] == pytest.approx(row["account_allocation"] + micro)
    assert result.micro_detail.select(pl.struct("period", "account", "sector").is_duplicated().any()).item() is False
    assert result.integrated_detail.height == 9


def test_methods_and_custom_input_column_names():
    original = positions().filter(pl.col("period") == "2026-01")
    mapping = {"period": "Month", "account": "Book", "sector": "Industry",
               "security": "Asset", "portfolio_value": "PMV", "benchmark_value": "BMV",
               "portfolio_return": "PR", "benchmark_return": "BR"}
    custom = original.rename(mapping)
    args = dict(column_period="Month", column_account="Book", column_sector="Industry",
                column_security="Asset", column_portfolio_value="PMV",
                column_benchmark_value="BMV", column_portfolio_return="PR",
                column_benchmark_return="BR")
    macro = macro_micro_attribution(custom, macro_attribution=bhb, micro_attribution=bhb,
                                    macro_effects=2, micro_effects=2,
                                    column_micro_segment="Asset", **args)
    pooled = value_weighted_total_attribution(custom, attribution=bhb, effects=2,
                                               column_segment="Asset", **args)
    decision = hierarchical_decision_attribution(custom, **args)
    assert macro.columns.period == "Month"
    assert {"Month", "Book", "Asset"}.issubset(macro.micro_detail.columns)
    assert macro.micro_detail["interaction"].sum() == pytest.approx(0)
    assert {"Month", "Asset"}.issubset(pooled.detail.columns)
    assert {"Month", "Book", "Industry", "Asset"}.issubset(decision.detail.columns)
    excesses = [macro.integrated_summary["excess_return"][0],
                pooled.summary["excess_return"][0], decision.summary["excess_return"][0]]
    assert excesses == pytest.approx([excesses[0]] * 3)


def test_rejects_undefined_conditionals_and_bad_positions():
    original = positions().filter(pl.col("period") == "2026-01")
    duplicate = pl.concat([original, original.head(1)])
    with pytest.raises(AttributionError, match="duplicate"):
        macro_micro_attribution(duplicate)
    with pytest.raises(AttributionError, match="nonnegative"):
        value_weighted_total_attribution(original.with_columns(
            pl.when(pl.col("account") == "Growth").then(-1.0)
            .otherwise(pl.col("portfolio_value")).alias("portfolio_value")))
    missing_sector = original.with_columns(
        pl.when((pl.col("account") == "Growth") & (pl.col("sector") == "Healthcare"))
        .then(0.0).otherwise(pl.col("benchmark_value")).alias("benchmark_value"))
    with pytest.raises(AttributionError, match="micro segment"):
        macro_micro_attribution(missing_sector)
    with pytest.raises(AttributionError, match="account/sector"):
        hierarchical_decision_attribution(missing_sector)


@pytest.mark.parametrize("linking", [link_carino, link_grap, link_frongello, link_menchero])
def test_arithmetic_linking_preserves_all_multi_account_channels(linking):
    data = positions()
    macro = macro_micro_attribution(data)
    decision = hierarchical_decision_attribution(data)
    pooled = value_weighted_total_attribution(data)
    first = link_macro_micro_attribution(macro, linking=linking)
    second = link_hierarchical_decision_attribution(decision, linking=linking)
    reference = linking(pooled).summary["excess_return"][0]
    assert first.linking_method == second.linking_method == linking.__name__
    for linked in (first, second):
        assert linked.summary["excess_return"][0] == pytest.approx(reference)
        assert linked.summary["total_effect"][0] == pytest.approx(reference)
        assert linked.detail["factor"].n_unique() == data["period"].n_unique()
    for row in first.detail.to_dicts():
        assert row["linked_account_allocation"] == pytest.approx(
            row["account_allocation"] * row["factor"])
    for row in second.detail.to_dicts():
        assert row["linked_effect"] == pytest.approx(row["effect"] * row["factor"])


def test_geometric_linking_is_explicitly_rejected_for_additive_stage_effects():
    data = positions()
    with pytest.raises(AttributionError, match="arithmetic linker"):
        link_macro_micro_attribution(macro_micro_attribution(data), linking=link_geometric)
    with pytest.raises(AttributionError, match="arithmetic linker"):
        link_hierarchical_decision_attribution(hierarchical_decision_attribution(data),
                                               linking=link_geometric)
