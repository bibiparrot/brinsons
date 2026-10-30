"""Explicit single-period and multiple-period Brinson calculation interfaces."""
from dataclasses import dataclass
from typing import Callable
import polars as pl

from .core import AttributionError, AttributionResult, _require, fachler, link_menchero
from .columns import ColumnNames


@dataclass(frozen=True)
class MultiplePeriodAttribution:
    """Period-by-period attribution and the separately linked horizon result."""
    periods: AttributionResult
    linked: AttributionResult


def _period_data(data: pl.DataFrame, column_period: str, period=None) -> pl.DataFrame:
    if not isinstance(column_period, str) or not column_period:
        raise AttributionError("column_period must be a nonempty string")
    _require(data, [column_period])
    if period is not None:
        data = data.filter(pl.col(column_period) == pl.lit(period))
        if data.is_empty():
            raise AttributionError(f"period {period!r} is not present")
    return data


def calculate_single_period(data: pl.DataFrame, *, period=None,
        attribution: Callable = fachler, effects: int = 3,
        column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        columns: ColumnNames | None = None) -> AttributionResult:
    """Calculate a single period using a named attribution function.

    Pass ``period`` to select one period from a multi-period frame; otherwise
    the input must contain exactly one period. ``attribution`` may be ``bhb``
    or ``fachler``, or a compatible callable. No multi-period linking is used.
    """
    if columns is not None and not isinstance(columns, ColumnNames):
        raise TypeError("columns must be a ColumnNames instance")
    period_name = columns.period if columns is not None else column_period
    data = _period_data(data, period_name, period)
    if data[period_name].n_unique() != 1:
        raise AttributionError("single-period calculation requires exactly one period; select period explicitly")
    if not callable(attribution):
        raise TypeError("attribution must be a callable such as bhb or fachler")
    result = attribution(data, effects=effects, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        **({"columns": columns} if columns is not None else {}))
    if not isinstance(result, AttributionResult) or result.summary.height != 1:
        raise AttributionError("attribution callable must return one-period AttributionResult")
    return result


def calculate_multiple_period(data: pl.DataFrame, *, attribution: Callable = fachler,
        linking: Callable = link_menchero, effects: int = 3,
        column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        columns: ColumnNames | None = None) -> MultiplePeriodAttribution:
    """Calculate each period and link them with a named linking function.

    Requires at least two periods. Returns ``periods`` (the per-period result)
    and ``linked`` (the full-horizon result). ``linking`` can be any of the
    exported linking functions or a compatible callable. Geometric linking uses
    a multiplicative active-return identity; the others use an additive one.
    """
    if columns is not None and not isinstance(columns, ColumnNames):
        raise TypeError("columns must be a ColumnNames instance")
    period_name = columns.period if columns is not None else column_period
    data = _period_data(data, period_name)
    count = data[period_name].n_unique()
    if count < 2:
        raise AttributionError("multiple-period calculation requires at least two periods")
    if not callable(attribution) or not callable(linking):
        raise TypeError("attribution and linking must be callable functions")
    periods = attribution(data, effects=effects, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        **({"columns": columns} if columns is not None else {}))
    if not isinstance(periods, AttributionResult) or periods.summary.height != count:
        raise AttributionError("attribution callable must return an AttributionResult for every period")
    linked = linking(periods)
    if not isinstance(linked, AttributionResult) or linked.summary.height != 1:
        raise AttributionError("linking callable must return a single-horizon AttributionResult")
    return MultiplePeriodAttribution(periods=periods, linked=linked)
