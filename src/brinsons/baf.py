"""Transparent Behavioral Alpha Frontier (BAF) research model.

This is an independent implementation inspired by the published EBAF
framework. Its formulas are specified in this project; the source paper
does not disclose every decision algorithm. Decision values are not
Brinson effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, isfinite, log
from typing import Iterator

import polars as pl


class BAFError(ValueError):
    """Invalid BAF input or parameter."""


@dataclass(frozen=True)
class BAFColumns:
    """Column roles for caller data and calculated decision fields."""

    portfolio: str = "portfolio"
    date: str = "date"
    security: str = "security"
    shares: str = "shares"
    price: str = "price"
    benchmark_level: str = "benchmark_level"
    episode: str = "episode"
    decision_type: str = "decision_type"
    value_added: str = "value_added"


@dataclass(frozen=True)
class BAFPrepared:
    """Validated daily grid and closed holding episodes."""

    daily: pl.DataFrame
    episodes: pl.DataFrame
    columns: BAFColumns


@dataclass(frozen=True)
class BAFResult:
    """Episode decisions and their descriptive frontier coordinates."""

    prepared: BAFPrepared
    decisions: pl.DataFrame
    metrics: pl.DataFrame
    payoff_scale_exponent: float = 1.5


_INTERNAL = ("_baf_active", "_baf_prev", "_baf_start", "_baf_exit",
             "_baf_equal_shares", "_baf_total_value", "_baf_active_count")


def prepare_baf(
    data: pl.DataFrame,
    *,
    column_portfolio: str = "portfolio",
    column_date: str = "date",
    column_security: str = "security",
    column_shares: str = "shares",
    column_price: str = "price",
    column_benchmark_level: str = "benchmark_level",
    column_episode: str = "episode",
    column_decision_type: str = "decision_type",
    column_value_added: str = "value_added",
    open_episode_policy: str = "exclude",
) -> BAFPrepared:
    """Find uninterrupted nonzero holdings from a full daily security grid.

    Shares are post-trade close-of-day holdings. Include one zero-share row
    on the liquidation date and subsequent prices for timing windows. Prices
    and shares must be split-adjusted consistently; dividends are excluded.
    Input names are retained; no source columns are renamed.
    """
    if not isinstance(data, pl.DataFrame):
        raise BAFError("BAF input must be a Polars DataFrame")
    c = BAFColumns(column_portfolio, column_date, column_security, column_shares,
                   column_price, column_benchmark_level, column_episode,
                   column_decision_type, column_value_added)
    source = [c.portfolio, c.date, c.security, c.shares, c.price, c.benchmark_level]
    if len(set(source + [c.episode, c.decision_type, c.value_added] + list(_INTERNAL))) != len(source) + 3 + len(_INTERNAL):
        raise BAFError("BAF input, output, and internal column names must be distinct")
    missing = [name for name in source if name not in data.columns]
    if missing:
        raise BAFError(f"Missing columns: {missing}")
    reserved = [name for name in [c.episode, c.decision_type, c.value_added, *_INTERNAL]
                if name in data.columns]
    if reserved:
        raise BAFError(f"Input contains reserved calculated columns: {reserved}")
    if open_episode_policy not in {"exclude", "error"}:
        raise BAFError("open_episode_policy must be 'exclude' or 'error'")
    if data.is_empty():
        raise BAFError("BAF input is empty")
    try:
        daily = data.with_columns(
            pl.col(c.date).cast(pl.Date),
            pl.col(c.shares).cast(pl.Float64),
            pl.col(c.price).cast(pl.Float64),
            pl.col(c.benchmark_level).cast(pl.Float64),
        ).sort([c.portfolio, c.security, c.date])
    except Exception as exc:
        raise BAFError("Could not cast date, shares, price, or benchmark level") from exc
    if daily.select(pl.any_horizontal([pl.col(name).is_null() for name in source]).any()).item():
        raise BAFError("BAF source columns may not contain nulls")
    if daily.filter((pl.col(c.shares) < 0) | (~pl.col(c.shares).is_finite()) |
                    (pl.col(c.price) <= 0) | (~pl.col(c.price).is_finite()) |
                    (pl.col(c.benchmark_level) <= 0) | (~pl.col(c.benchmark_level).is_finite())).height:
        raise BAFError("Shares must be finite and nonnegative; price and benchmark level finite and positive")
    if daily.group_by([c.portfolio, c.security, c.date]).len().filter(pl.col("len") != 1).height:
        raise BAFError("Each portfolio/security/date must occur once")
    if daily.group_by([c.portfolio, c.date]).agg(pl.col(c.benchmark_level).n_unique().alias("_n")).filter(pl.col("_n") != 1).height:
        raise BAFError("Benchmark level must agree within each portfolio/date")
    daily = daily.with_columns(
        (pl.col(c.shares) > 0).alias("_baf_active"),
        (pl.col(c.shares).shift(1).over([c.portfolio, c.security]).fill_null(0) > 0).alias("_baf_prev"),
    ).with_columns(
        (pl.col("_baf_active") & ~pl.col("_baf_prev")).alias("_baf_start"),
        (~pl.col("_baf_active") & pl.col("_baf_prev")).alias("_baf_exit"),
    ).with_columns(
        pl.col("_baf_start").cast(pl.UInt32).cum_sum().over([c.portfolio, c.security]).alias(c.episode),
        pl.when(pl.col("_baf_active"))
        .then(pl.col(c.shares) * pl.col(c.price))
        .otherwise(0.0).sum().over([c.portfolio, c.date]).alias("_baf_total_value"),
        pl.col("_baf_active").cast(pl.UInt32).sum().over([c.portfolio, c.date]).alias("_baf_active_count"),
    ).with_columns(
        pl.when(pl.col("_baf_active"))
        .then(pl.col("_baf_total_value") / pl.col("_baf_active_count") / pl.col(c.price))
        .otherwise(0.0).alias("_baf_equal_shares")
    ).drop("_baf_total_value", "_baf_active_count")
    active = daily.filter(pl.col("_baf_active"))
    episodes = active.group_by([c.portfolio, c.security, c.episode]).agg(
        pl.col(c.date).min().alias("entry_date"),
        pl.col(c.date).max().alias("last_active_date"),
        pl.col(c.shares).max().alias("peak_shares"),
        pl.len().alias("active_observations"),
    )
    exits = daily.filter(pl.col("_baf_exit")).select(
        c.portfolio, c.security, c.episode, pl.col(c.date).alias("exit_date")
    )
    episodes = episodes.join(exits, on=[c.portfolio, c.security, c.episode], how="left")
    if open_episode_policy == "error" and episodes.filter(pl.col("exit_date").is_null()).height:
        raise BAFError("At least one episode has no observed zero-share exit row")
    episodes = episodes.filter(pl.col("exit_date").is_not_null()).sort([c.portfolio, c.security, c.episode])
    return BAFPrepared(daily, episodes, c)


def _paths(prepared: BAFPrepared) -> Iterator[tuple[dict, list[dict], int, int]]:
    c = prepared.columns
    for ep in prepared.episodes.iter_rows(named=True):
        rows = prepared.daily.filter(
            (pl.col(c.portfolio) == ep[c.portfolio]) &
            (pl.col(c.security) == ep[c.security])
        ).sort(c.date).iter_rows(named=True)
        path = list(rows)
        start = next(i for i, row in enumerate(path) if row[c.date] == ep["entry_date"])
        end = next(i for i, row in enumerate(path) if row[c.date] == ep["exit_date"])
        yield ep, path, start, end


def _decision_frame(prepared: BAFPrepared, kind: str, values: list[tuple[dict, float]]) -> pl.DataFrame:
    c = prepared.columns
    schema = {c.portfolio: prepared.daily.schema[c.portfolio],
              c.security: prepared.daily.schema[c.security], c.episode: pl.UInt32,
              c.decision_type: pl.String, c.value_added: pl.Float64}
    return pl.DataFrame([
        {c.portfolio: ep[c.portfolio], c.security: ep[c.security], c.episode: ep[c.episode],
         c.decision_type: kind, c.value_added: float(value)}
        for ep, value in values if isfinite(value)
    ], schema=schema)


def _relative(row: dict, c: BAFColumns) -> float:
    return row[c.price] / row[c.benchmark_level]


def _money_weighted_return(rows: list[dict], shares_key: str, c: BAFColumns) -> float | None:
    if len(rows) < 2:
        return None
    first, last = rows[0][c.date], rows[-1][c.date]
    duration = (last - first).days
    if duration <= 0:
        return None
    previous = 0.0
    flows = []
    for row in rows:
        current = row[shares_key]
        flows.append((-(current - previous) * row[c.price], (row[c.date] - first).days / duration))
        previous = current
    if previous != 0.0:
        return None
    if not any(flow < 0 for flow, _ in flows) or not any(flow > 0 for flow, _ in flows):
        return None

    def npv(log_gross: float) -> float:
        return sum(flow * exp(-log_gross * fraction) for flow, fraction in flows)

    low, high = -20.0, 20.0
    if npv(low) * npv(high) > 0:
        return None
    for _ in range(100):
        middle = (low + high) / 2
        if npv(low) * npv(middle) <= 0:
            high = middle
        else:
            low = middle
    return exp((low + high) / 2) - 1


def stock_picking_decisions(prepared: BAFPrepared) -> pl.DataFrame:
    """Published description: instrument TWR minus nominated benchmark TWR."""
    c = prepared.columns
    return _decision_frame(prepared, "stock_picking", [
        (ep, path[end][c.price] / path[start][c.price]
         - path[end][c.benchmark_level] / path[start][c.benchmark_level])
        for ep, path, start, end in _paths(prepared)
    ])


def sizing_decisions(prepared: BAFPrepared) -> pl.DataFrame:
    """Published baseline: actual episode MWR minus equal-active-weight MWR.

    The counterfactual rebalances the portfolio's invested market value
    equally among its active securities at each observed close. The paper
    does not disclose its precise cash-flow or rebalancing convention.
    """
    c = prepared.columns
    values = []
    for ep, path, start, end in _paths(prepared):
        rows = path[start:end + 1]
        actual = _money_weighted_return(rows, c.shares, c)
        equal = _money_weighted_return(rows, "_baf_equal_shares", c)
        if actual is not None and equal is not None:
            values.append((ep, actual - equal))
    return _decision_frame(prepared, "sizing", values)


def entry_timing_decisions(prepared: BAFPrepared, *, forward_days: int = 21) -> pl.DataFrame:
    """Mean next-N relative prices versus relative price at entry."""
    if forward_days < 1:
        raise BAFError("forward_days must be positive")
    c = prepared.columns
    values = []
    for ep, path, start, _ in _paths(prepared):
        if start + forward_days < len(path):
            future = sum(_relative(row, c) for row in path[start + 1:start + 1 + forward_days]) / forward_days
            values.append((ep, future / _relative(path[start], c) - 1))
    return _decision_frame(prepared, "entry_timing", values)


def _phase(path: list[dict], start: int, end: int, c: BAFColumns, fraction: float) -> tuple[int, int]:
    if not 0 < fraction <= 1:
        raise BAFError("phase_fraction must lie in (0, 1]")
    peak = max(row[c.shares] for row in path[start:end])
    build_end = next(i for i in range(start, end) if path[i][c.shares] >= fraction * peak)
    # Exit phase begins at the first below-threshold holding that never recovers.
    exit_start = end
    for i in range(build_end + 1, end):
        if path[i][c.shares] < fraction * peak and all(
            path[j][c.shares] < fraction * peak for j in range(i, end)
        ):
            exit_start = i
            break
    return build_end, exit_start


def scaling_in_decisions(prepared: BAFPrepared, *, phase_fraction: float = 0.9) -> pl.DataFrame:
    """Actual build cost versus buying build-complete shares at entry."""
    if not 0 < phase_fraction <= 1:
        raise BAFError("phase_fraction must lie in (0, 1]")
    c = prepared.columns
    values = []
    for ep, path, start, end in _paths(prepared):
        build_end, _ = _phase(path, start, end, c, phase_fraction)
        if build_end == start:
            continue
        changes = [(path[i][c.shares] - (path[i - 1][c.shares] if i > start else 0.0),
                    _relative(path[i], c)) for i in range(start, build_end + 1)]
        if any(change < 0 for change, _ in changes):
            continue
        target = path[build_end][c.shares]
        baseline = target * _relative(path[start], c)
        values.append((ep, (baseline - sum(change * relative for change, relative in changes)) / baseline))
    return _decision_frame(prepared, "scaling_in", values)


def size_adjusting_decisions(prepared: BAFPrepared, *, phase_fraction: float = 0.9) -> pl.DataFrame:
    """Middle-phase variable shares versus constant build-complete shares."""
    if not 0 < phase_fraction <= 1:
        raise BAFError("phase_fraction must lie in (0, 1]")
    c = prepared.columns
    values = []
    for ep, path, start, end in _paths(prepared):
        build_end, exit_start = _phase(path, start, end, c, phase_fraction)
        if exit_start - build_end < 2:
            continue
        base = path[build_end][c.shares]
        if not any(path[i][c.shares] != base for i in range(build_end + 1, exit_start)):
            continue
        value = sum((path[i][c.shares] - base) *
                    (_relative(path[i + 1], c) - _relative(path[i], c))
                    for i in range(build_end, exit_start - 1)) / (base * _relative(path[build_end], c))
        values.append((ep, value))
    return _decision_frame(prepared, "size_adjusting", values)


def scaling_out_decisions(prepared: BAFPrepared, *, phase_fraction: float = 0.9) -> pl.DataFrame:
    """Gradual exit P&L versus full liquidation at first persistent cut."""
    if not 0 < phase_fraction <= 1:
        raise BAFError("phase_fraction must lie in (0, 1]")
    c = prepared.columns
    values = []
    for ep, path, start, end in _paths(prepared):
        _, exit_start = _phase(path, start, end, c, phase_fraction)
        if exit_start >= end or exit_start == start:
            continue
        before = path[exit_start - 1][c.shares]
        value = sum(path[i][c.shares] * (_relative(path[i + 1], c) - _relative(path[i], c))
                    for i in range(exit_start, end)) / (before * _relative(path[exit_start], c))
        values.append((ep, value))
    return _decision_frame(prepared, "scaling_out", values)


def exit_timing_decisions(prepared: BAFPrepared, *, forward_days: int = 21) -> pl.DataFrame:
    """Relative price at exit versus mean next-N relative prices.

    The source paper uses a fixed one-month evaluation after exit, but does
    not disclose the formula. ``forward_days=21`` approximates that window.
    """
    if forward_days < 1:
        raise BAFError("forward_days must be positive")
    c = prepared.columns
    values = []
    for ep, path, _, end in _paths(prepared):
        if end + forward_days < len(path):
            future = sum(_relative(row, c) for row in path[end + 1:end + 1 + forward_days]) / forward_days
            values.append((ep, 1 - future / _relative(path[end], c)))
    return _decision_frame(prepared, "exit_timing", values)


def payoff_scale(payoff: float, *, exponent: float = 1.5) -> float:
    """Illustrative log-logistic scale: ``P**k / (1 + P**k)``.

    The paper does not disclose its scale. The default ``k=1.5`` is an
    illustrative shape choice, not a fitted or published parameter.
    """
    if not isfinite(exponent) or exponent <= 0:
        raise BAFError("exponent must be finite and positive")
    if not isfinite(payoff) or payoff < 0:
        raise BAFError("payoff must be finite and nonnegative")
    if payoff == 0:
        return 0.0
    z = exponent * log(payoff)
    if z >= 0:
        return 1 / (1 + exp(-z))
    ez = exp(z)
    return ez / (1 + ez)


def inverse_payoff_scale(scaled_payoff: float, *, exponent: float = 1.5) -> float:
    """Recover payoff from an interior scaled coordinate under this model."""
    if not isfinite(exponent) or exponent <= 0:
        raise BAFError("exponent must be finite and positive")
    if not isfinite(scaled_payoff) or not 0 < scaled_payoff < 1:
        raise BAFError("scaled_payoff must lie strictly between zero and one")
    return exp((log(scaled_payoff) - log(1 - scaled_payoff)) / exponent)


def normalized_edge(hit_rate: float, payoff: float) -> float:
    """Mean value added divided by mean absolute loss: ``h*P-(1-h)``.

    The value is a normalized ratio, not raw portfolio or decision return.
    """
    if not isfinite(hit_rate) or not 0 <= hit_rate <= 1:
        raise BAFError("hit_rate must lie between zero and one")
    if not isfinite(payoff) or payoff < 0:
        raise BAFError("payoff must be finite and nonnegative")
    return hit_rate * payoff - (1 - hit_rate)


def decision_metrics(decisions: pl.DataFrame, *, columns: BAFColumns = BAFColumns(),
                     minimum_decisions: int = 1, include_all: bool = True,
                     payoff_scale_exponent: float = 1.5) -> pl.DataFrame:
    """Hit rate and payoff from the paper; explicitly chosen scaled payoff.

    Zero decisions are wins. Payoff is null if no negative or no nonnegative
    observations exist, so frontier position is then undefined. The assumed
    scale ``P**k/(1+P**k)`` is not Essentia's undisclosed transform.
    """
    if minimum_decisions < 1:
        raise BAFError("minimum_decisions must be positive")
    if not isfinite(payoff_scale_exponent) or payoff_scale_exponent <= 0:
        raise BAFError("payoff_scale_exponent must be finite and positive")
    c = columns
    needed = [c.portfolio, c.decision_type, c.value_added]
    if any(name not in decisions.columns for name in needed):
        raise BAFError(f"Decisions require columns {needed}")
    if decisions.filter(pl.col(c.value_added).is_null() | ~pl.col(c.value_added).is_finite()).height:
        raise BAFError("Decision values must be finite")
    data = decisions
    if include_all:
        combined = decisions.with_columns(pl.lit("all_decisions").alias(c.decision_type))
        data = pl.concat([decisions, combined])
    return (data.group_by([c.portfolio, c.decision_type]).agg(
        pl.len().alias("n_decisions"),
        (pl.col(c.value_added) >= 0).sum().alias("n_hits"),
        (pl.col(c.value_added) < 0).sum().alias("n_losses"),
        pl.col(c.value_added).filter(pl.col(c.value_added) >= 0).mean().alias("mean_hit"),
        pl.col(c.value_added).filter(pl.col(c.value_added) < 0).mean().alias("mean_loss"),
        pl.col(c.value_added).mean().alias("mean_value_added"),
    ).filter(pl.col("n_decisions") >= minimum_decisions).with_columns(
        (pl.col("n_hits") / pl.col("n_decisions")).alias("hit_rate"),
        pl.when((pl.col("n_hits") > 0) & (pl.col("n_losses") > 0))
        .then(pl.col("mean_hit") / pl.col("mean_loss").abs())
        .otherwise(None).alias("payoff"),
    ).with_columns(
        (1 / (1 + (-payoff_scale_exponent * pl.col("payoff").log()).exp())).alias("scaled_payoff")
    ).sort([c.decision_type, c.portfolio]))


def calculate_baf(
    prepared: BAFPrepared,
    *,
    window_start: date | None = None,
    window_end: date | None = None,
    entry_forward_days: int = 21,
    exit_forward_days: int = 21,
    scaling_in_fraction: float = 0.9,
    size_adjusting_fraction: float = 0.9,
    scaling_out_fraction: float = 0.9,
    minimum_decisions: int = 1,
    payoff_scale_exponent: float = 1.5,
) -> BAFResult:
    """Run named tools on completed episodes wholly inside an optional window.

    Window selection uses episode entry and exit dates. The original daily
    grid is kept for post-exit timing. This is a descriptive completed-episode
    policy, not the source paper's proprietary quarterly eligibility rule.
    """
    if window_start is not None and window_end is not None and window_start > window_end:
        raise BAFError("window_start must be on or before window_end")
    if window_start is not None or window_end is not None:
        episodes = prepared.episodes
        if window_start is not None:
            episodes = episodes.filter(pl.col("entry_date") >= window_start)
        if window_end is not None:
            episodes = episodes.filter(pl.col("exit_date") <= window_end)
        prepared = BAFPrepared(prepared.daily, episodes, prepared.columns)
    frames = [
        stock_picking_decisions(prepared),
        sizing_decisions(prepared),
        entry_timing_decisions(prepared, forward_days=entry_forward_days),
        scaling_in_decisions(prepared, phase_fraction=scaling_in_fraction),
        size_adjusting_decisions(prepared, phase_fraction=size_adjusting_fraction),
        scaling_out_decisions(prepared, phase_fraction=scaling_out_fraction),
        exit_timing_decisions(prepared, forward_days=exit_forward_days),
    ]
    decisions = pl.concat(frames)
    metrics = decision_metrics(decisions, columns=prepared.columns,
                               minimum_decisions=minimum_decisions,
                               payoff_scale_exponent=payoff_scale_exponent)
    return BAFResult(prepared, decisions, metrics, payoff_scale_exponent)


def plot_baf(
    result: BAFResult,
    *,
    portfolio: str | None = None,
    figsize: tuple[float, float] = (16, 9),
    payoff_scale_exponent: float | None = None,
    contour_levels: tuple[float, ...] = (-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 5, 50),
    grid_size: int = 161,
    cmap: str = "Purples",
):
    """Plot eight BAF panels with data-independent normalized-edge contours.

    Color bands show ``E=h*P-(1-h)`` after inverting this package's assumed
    log-logistic payoff scale. The thick dashed ``E=0`` curve is the model's
    chance frontier. Neither the exponent nor band levels are published by
    Essentia. A shared colorbar describes the normalized edge; scatter points
    are plotted on top. Returns Matplotlib ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    exponent = result.payoff_scale_exponent if payoff_scale_exponent is None else payoff_scale_exponent
    if not isfinite(exponent) or exponent <= 0:
        raise BAFError("payoff_scale_exponent must be finite and positive")
    if grid_size < 20:
        raise BAFError("grid_size must be at least 20")
    if len(contour_levels) < 2 or any(not isfinite(level) for level in contour_levels) or any(
        right <= left for left, right in zip(contour_levels, contour_levels[1:])
    ):
        raise BAFError("contour_levels must contain at least two increasing finite values")

    c = result.prepared.columns
    labels = ["stock_picking", "entry_timing", "sizing", "scaling_in",
              "size_adjusting", "scaling_out", "exit_timing", "all_decisions"]
    frame = result.metrics
    if portfolio is not None:
        frame = frame.filter(pl.col(c.portfolio) == portfolio)
    epsilon = 1e-4
    coordinates = [epsilon + (1 - 2 * epsilon) * i / (grid_size - 1) for i in range(grid_size)]
    inverse = [inverse_payoff_scale(s, exponent=exponent) for s in coordinates]
    edge_grid = [[normalized_edge(h, payoff) for h in coordinates] for payoff in inverse]
    chance = [payoff_scale((1 - h) / h, exponent=exponent) for h in coordinates]

    fig, axes = plt.subplots(2, 4, figsize=figsize, squeeze=False, constrained_layout=True)
    norm = BoundaryNorm(contour_levels, ncolors=plt.get_cmap(cmap).N, extend="both")
    filled = None
    for ax, label in zip(axes.flat, labels):
        filled = ax.contourf(coordinates, coordinates, edge_grid, levels=contour_levels,
                             norm=norm, cmap=cmap, extend="both", alpha=0.84)
        boundary_levels = [level for level in contour_levels[1:-1] if level != 0]
        if boundary_levels:
            ax.contour(coordinates, coordinates, edge_grid,
                       levels=boundary_levels, colors="white",
                       linewidths=0.45, alpha=0.45)
        ax.plot(coordinates, chance, color="black", linestyle="--", linewidth=2,
                zorder=3, label="E=0 (chance)")
        sub = frame.filter((pl.col(c.decision_type) == label) & pl.col("payoff").is_not_null())
        for row in sub.iter_rows(named=True):
            payoff = row["payoff"]
            if not isfinite(payoff) or payoff < 0:
                continue
            scaled = payoff_scale(payoff, exponent=exponent)
            ax.scatter(row["hit_rate"], scaled, s=45, color="#c43c39",
                       edgecolors="white", linewidths=0.7, zorder=4)
            ax.annotate(str(row[c.portfolio]), (row["hit_rate"], scaled), fontsize=7,
                        xytext=(3, 3), textcoords="offset points", zorder=5)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=label.replace("_", " "),
               xlabel="Hit rate", ylabel=f"Scaled payoff (k={exponent:g})")
        ax.grid(alpha=0.12)
    axes[0, 0].legend(loc="lower left", fontsize=7)
    if filled is not None:
        fig.colorbar(filled, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02,
                     label="Normalized edge E = hP − (1−h) (illustrative)")
    fig.suptitle("Behavioral Alpha Frontier")
    return fig, axes


def baf_change_table(previous: BAFResult, current: BAFResult) -> pl.DataFrame:
    """Compare two BAF windows across common peers, displaying changes as %.

    Inspired by the *layout* of the paper's Exhibit 8, this function reports
    changes in the share of portfolios with (1) positive mean decision value,
    (2) hit rate above 50%, and (3) payoff above one. It uses this package's
    independently defined decision values, not Essentia's Decision Alpha.
    Only portfolio/decision pairs present in both results with defined payoff
    in both windows enter the denominator. This paired-cohort rule can differ
    from the paper's cohort and eligibility rules.
    """
    c = previous.prepared.columns
    other = current.prepared.columns
    if (c.portfolio, c.decision_type) != (other.portfolio, other.decision_type):
        raise BAFError("BAF results must use the same portfolio and decision-type column names")
    keys = [c.portfolio, c.decision_type]
    paired = (previous.metrics.join(current.metrics, on=keys, how="inner", suffix="_current")
              .filter(pl.col("payoff").is_not_null() & pl.col("payoff_current").is_not_null()))
    if paired.is_empty():
        raise BAFError("No common portfolio/decision pairs with defined payoff in both windows")
    changes = paired.group_by(c.decision_type).agg(
        pl.len().alias("n_common_portfolios"),
        (100 * ((pl.col("mean_value_added_current") > 0).mean() -
                (pl.col("mean_value_added") > 0).mean())).alias("adding_value_change_percent"),
        (100 * ((pl.col("hit_rate_current") > 0.5).mean() -
                (pl.col("hit_rate") > 0.5).mean())).alias("hit_rate_positive_change_percent"),
        (100 * ((pl.col("payoff_current") > 1).mean() -
                (pl.col("payoff") > 1).mean())).alias("payoff_positive_change_percent"),
    )
    order = ["all_decisions", "entry_timing", "exit_timing",
             "scaling_in", "scaling_out", "size_adjusting",
             "sizing", "stock_picking"]
    sequence = pl.DataFrame({c.decision_type: order, "_baf_display_order": list(range(len(order)))})
    return (changes.join(sequence, on=c.decision_type, how="left")
            .sort("_baf_display_order", c.decision_type).drop("_baf_display_order"))


def chart_baf_change_table(
    changes: pl.DataFrame,
    *,
    column_decision_type: str = "decision_type",
    column_adding_value_change: str = "adding_value_change_percent",
    column_hit_rate_change: str = "hit_rate_positive_change_percent",
    column_payoff_change: str = "payoff_positive_change_percent",
    title: str = "BAF peer changes: current − previous window",
    figsize: tuple[float, float] = (10, 5),
):
    """Draw a red/green Exhibit-8-style table of BAF percentage differences.

    Green means a larger share of the paired cohort passes the indicated
    condition; pink means a smaller share. The chart is not a reproduction
    of the paper's values or its proprietary Decision Alpha calculation.
    Returns Matplotlib ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt

    names = [column_decision_type, column_adding_value_change,
             column_hit_rate_change, column_payoff_change]
    missing = [name for name in names if name not in changes.columns]
    if missing:
        raise BAFError(f"Missing table columns: {missing}")
    if changes.is_empty():
        raise BAFError("Change table is empty")
    if changes.select(pl.any_horizontal([pl.col(name).is_null() for name in names[1:]]).any()).item():
        raise BAFError("Change table contains null percentage values")
    labels = {
        "all_decisions": "All decisions",
        "entry_timing": "Entry",
        "exit_timing": "Exit",
        "scaling_in": "Scaling in",
        "scaling_out": "Scaling out",
        "size_adjusting": "Size adjusting",
        "sizing": "Sizing",
        "stock_picking": "Stock picking",
    }
    rows = changes.select(names).iter_rows(named=True)
    values = []
    for row in rows:
        values.append([labels.get(str(row[column_decision_type]), str(row[column_decision_type]))] +
                      [f"{row[name]:+.1f}%" for name in names[1:]])
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, loc="left", color="#344b98", fontweight="bold", fontsize=15, pad=20)
    table = ax.table(
        cellText=values,
        colLabels=["Decision type", "Adding value", "Hit rate >50%", "Payoff >1"],
        cellLoc="center", colLoc="center", colWidths=[0.34, 0.22, 0.22, 0.22],
        bbox=[0.0, 0.08, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row_index == 0:
            cell.set_facecolor("#eef1f8")
            cell.set_text_props(weight="bold", color="#24345f")
        elif column_index == 0:
            cell.set_facecolor("white")
            cell.set_text_props(ha="left")
        else:
            value = changes[names[column_index]][row_index - 1]
            cell.set_facecolor("#bde1d0" if value > 0 else "#f3caca" if value < 0 else "#eeeeee")
    ax.text(0, 0.025, "Current percentage minus previous percentage; common eligible portfolios.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    return fig, ax
