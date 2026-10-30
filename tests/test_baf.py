from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from brinsons import (
    BAFError, calculate_baf, decision_metrics, entry_timing_decisions,
    exit_timing_decisions, plot_baf, prepare_baf, sizing_decisions,
    stock_picking_decisions, scaling_in_decisions, size_adjusting_decisions,
    scaling_out_decisions, payoff_scale, inverse_payoff_scale, normalized_edge,
    baf_change_table, chart_baf_change_table,
)


def small_grid():
    prices = [100, 102, 104, 106, 108, 110, 112, 114, 116]
    shares = [0, 10, 10, 0, 0, 10, 10, 0, 0]
    return pl.DataFrame({
        "Account": ["A"] * len(prices),
        "Day": [date(2025, 1, 1) + timedelta(days=i) for i in range(len(prices))],
        "Asset": ["X"] * len(prices),
        "Units": shares,
        "Px": prices,
        "Index": [100.0] * len(prices),
    })


def prepared_small():
    return prepare_baf(small_grid(), column_portfolio="Account", column_date="Day",
                       column_security="Asset", column_shares="Units", column_price="Px",
                       column_benchmark_level="Index")


def test_episode_restart_and_source_names_retained():
    prepared = prepared_small()
    assert prepared.episodes.height == 2
    assert prepared.episodes["episode"].to_list() == [1, 2]
    assert set(small_grid().columns).issubset(prepared.daily.columns)
    picking = stock_picking_decisions(prepared)
    assert picking["value_added"].to_list() == pytest.approx([106 / 102 - 1, 114 / 110 - 1])
    assert sizing_decisions(prepared)["value_added"].to_list() == pytest.approx([0, 0], abs=1e-10)


def test_timing_and_window_filter():
    prepared = prepared_small()
    entry = entry_timing_decisions(prepared, forward_days=2)
    assert entry["value_added"][0] == pytest.approx((104 + 106) / 2 / 102 - 1)
    exit_ = exit_timing_decisions(prepared, forward_days=2)
    assert exit_.height == 1  # second exit lacks two future closes
    assert exit_["value_added"][0] == pytest.approx(1 - (108 + 110) / 2 / 106)
    run = calculate_baf(prepared, window_start=date(2025, 1, 5))
    assert run.prepared.episodes.height == 1
    assert run.decisions["episode"].unique().to_list() == [2]


def test_hit_payoff_and_undefined_payoff():
    prepared = prepared_small()
    c = prepared.columns
    decisions = pl.DataFrame({"Account": ["A"] * 4, "decision_type": ["test"] * 4,
                              "value_added": [-2.0, -1.0, 0.0, 3.0]})
    metric = decision_metrics(decisions, columns=c, include_all=False).row(0, named=True)
    assert metric["hit_rate"] == pytest.approx(0.5)
    assert metric["payoff"] == pytest.approx(1.0)
    assert metric["scaled_payoff"] == pytest.approx(0.5)
    only_wins = decisions.filter(pl.col("value_added") >= 0)
    assert decision_metrics(only_wins, columns=c)["payoff"].null_count() == 2


def test_normalized_edge_contours_and_scale_agree():
    assert normalized_edge(0.25, 3.0) == pytest.approx(0)
    curved = payoff_scale(3.0, exponent=1.5)
    assert curved == pytest.approx(0.838609522, abs=1e-8)
    assert curved > payoff_scale(3.0, exponent=1.0)
    assert inverse_payoff_scale(curved, exponent=1.5) == pytest.approx(3.0)
    decisions = pl.DataFrame({"portfolio": ["A"] * 4,
                              "decision_type": ["example"] * 4,
                              "value_added": [-1.0, -1.0, -1.0, 3.0]})
    metric = decision_metrics(decisions, include_all=False,
                              payoff_scale_exponent=1.5).row(0, named=True)
    assert metric["hit_rate"] == pytest.approx(0.25)
    assert metric["payoff"] == pytest.approx(3.0)
    assert metric["scaled_payoff"] == pytest.approx(curved)
    with pytest.raises(BAFError, match="exponent"):
        payoff_scale(3.0, exponent=0)


def test_three_phase_equations():
    shares = [0, 40, 40, 70, 100, 100, 92, 100, 70, 40, 0, 0]
    prices = [float(100 + i) for i in range(len(shares))]
    grid = pl.DataFrame({
        "portfolio": ["A"] * len(shares),
        "date": [date(2025, 1, 1) + timedelta(days=i) for i in range(len(shares))],
        "security": ["X"] * len(shares), "shares": shares,
        "price": prices, "benchmark_level": [100.0] * len(shares),
    })
    prepared = prepare_baf(grid)
    scaling_in = scaling_in_decisions(prepared)["value_added"][0]
    adjusting = size_adjusting_decisions(prepared)["value_added"][0]
    scaling_out = scaling_out_decisions(prepared)["value_added"][0]
    assert scaling_in == pytest.approx((100 * 101 - 40 * 101 - 30 * 103 - 30 * 104) / (100 * 101))
    assert adjusting == pytest.approx((-8 * (107 - 106)) / (100 * 104))
    assert scaling_out == pytest.approx((70 * (109 - 108) + 40 * (110 - 109)) / (100 * 108))


def test_validation_rejects_bad_inputs():
    with pytest.raises(BAFError, match="once"):
        prepare_baf(pl.concat([small_grid(), small_grid().head(1)]),
                    column_portfolio="Account", column_date="Day",
                    column_security="Asset", column_shares="Units", column_price="Px",
                    column_benchmark_level="Index")
    with pytest.raises(BAFError, match="positive"):
        entry_timing_decisions(prepared_small(), forward_days=0)


def test_synthetic_pipeline_and_chart():
    csv = Path(__file__).resolve().parents[1] / "data" / "baf" / "synthetic_daily_positions.csv"
    prepared = prepare_baf(pl.read_csv(csv, try_parse_dates=True))
    run = calculate_baf(prepared)
    assert prepared.episodes.height == 48
    assert run.decisions.height == 48 * 7
    assert run.metrics.height == 3 * 8
    assert run.metrics["scaled_payoff"].null_count() == 0
    pytest.importorskip("matplotlib")
    fig, axes = plot_baf(run)
    assert axes.shape == (2, 4)
    assert len(fig.axes) == 9  # eight panels plus the shared contour colorbar
    chance = axes[0, 0].lines[0]
    x, y = chance.get_xdata(), chance.get_ydata()
    closest = min(range(len(x)), key=lambda i: abs(x[i] - 0.25))
    assert y[closest] == pytest.approx(payoff_scale(3.0, exponent=1.5), abs=0.005)
    assert len(axes[0, 0].collections) >= 3  # filled bands, boundaries, manager dots
    import matplotlib.pyplot as plt
    plt.close(fig)
    overridden, other_axes = plot_baf(run, payoff_scale_exponent=1.0)
    other_chance = other_axes[0, 0].lines[0]
    other_x, other_y = other_chance.get_xdata(), other_chance.get_ydata()
    closest = min(range(len(other_x)), key=lambda i: abs(other_x[i] - 0.25))
    assert other_y[closest] == pytest.approx(0.75, abs=0.005)
    plt.close(overridden)


def test_peer_change_table_and_visual():
    csv = Path(__file__).resolve().parents[1] / "data" / "baf" / "synthetic_daily_positions.csv"
    prepared = prepare_baf(pl.read_csv(csv, try_parse_dates=True))
    early = calculate_baf(prepared, window_end=date(2025, 4, 1))
    late = calculate_baf(prepared, window_start=date(2025, 4, 1))
    changes = baf_change_table(early, late)
    assert changes.height == 8
    assert changes["n_common_portfolios"].to_list() == [3] * 8
    entry = changes.filter(pl.col("decision_type") == "entry_timing").row(0, named=True)
    assert entry["adding_value_change_percent"] == pytest.approx(100 / 3)
    assert entry["hit_rate_positive_change_percent"] == pytest.approx(100 / 3)
    assert entry["payoff_positive_change_percent"] == pytest.approx(200 / 3)
    pytest.importorskip("matplotlib")
    fig, ax = chart_baf_change_table(changes)
    assert len(ax.tables) == 1
    # Entry rises on all three measures; the next row (exit) falls.
    cells = ax.tables[0].get_celld()
    assert cells[2, 1].get_facecolor() != cells[3, 1].get_facecolor()
    import matplotlib.pyplot as plt
    plt.close(fig)
