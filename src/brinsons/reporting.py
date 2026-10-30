"""Markdown tables and Matplotlib charts for Brinson results."""
from __future__ import annotations

import polars as pl
from dataclasses import replace
from typing import Callable
from .columns import ColumnNames
from .core import (AttributionError, AttributionResult, _check, _require, _validate,
                   link_carino, link_frongello, link_geometric, link_grap, link_menchero)


def _inputs(result, column_period, column_segment, column_portfolio_weight,
            column_benchmark_weight, column_portfolio_return, column_benchmark_return,
            column_excess_return, column_allocation, column_selection, column_interaction):
    if not isinstance(result, AttributionResult):
        raise TypeError("expected an AttributionResult")
    requested = (column_period,column_segment,column_portfolio_weight,column_benchmark_weight,column_portfolio_return,
        column_benchmark_return,column_excess_return,column_allocation,column_selection,column_interaction)
    defaults = ("period","segment","portfolio_weight","benchmark_weight","portfolio_return","benchmark_return",
        "excess_return","allocation","selection","interaction")
    c = result.columns
    if requested != defaults:
        c = replace(c, period=column_period, segment=column_segment, portfolio_weight=column_portfolio_weight,
            benchmark_weight=column_benchmark_weight, portfolio_return=column_portfolio_return,
            benchmark_return=column_benchmark_return, allocation=column_allocation, selection=column_selection,
            interaction=column_interaction, summary_portfolio_return=column_portfolio_return,
            summary_benchmark_return=column_benchmark_return, summary_excess_return=column_excess_return,
            summary_allocation=column_allocation, summary_selection=column_selection, summary_interaction=column_interaction)
    effects = [c.allocation,c.selection,c.interaction]
    totals = [c.summary_allocation,c.summary_selection,c.summary_interaction]
    returns = [c.portfolio_return,c.benchmark_return]
    summary_returns = [c.summary_portfolio_return,c.summary_benchmark_return]
    detail = _validate(result.detail, [c.portfolio_weight,c.benchmark_weight,*returns,*effects], c)
    summary = result.summary
    numbers = [*summary_returns,c.summary_excess_return,*totals]
    _require(summary, [c.period,*numbers])
    summary = summary.with_columns(pl.col(list(dict.fromkeys(numbers))).cast(pl.Float64))
    if summary[c.period].n_unique() != summary.height or summary.select(pl.any_horizontal(~pl.col(x).is_finite() for x in numbers).any()).item():
        raise AttributionError("summary periods must be unique and numeric values finite")
    actual = detail.group_by(c.period).agg(
        (pl.col(c.portfolio_weight)*pl.col(c.portfolio_return)).sum().alias("_report_p"),
        (pl.col(c.benchmark_weight)*pl.col(c.benchmark_return)).sum().alias("_report_b"),
        *(pl.col(name).sum().alias("_report_effect_"+str(i)) for i,name in enumerate(effects)))
    paired = summary.join(actual, on=c.period, how="full", coalesce=True)
    for name, check in zip([*summary_returns,*totals],["_report_p","_report_b","_report_effect_0","_report_effect_1","_report_effect_2"]):
        if paired.select((pl.col(name).is_null() | pl.col(check).is_null()).any()).item():
            raise AttributionError("detail and summary periods differ")
        _check(paired, pl.col(name)-pl.col(check))
    _check(summary, pl.col(summary_returns[0])-pl.col(summary_returns[1])-pl.col(c.summary_excess_return))
    _check(summary, pl.sum_horizontal(totals)-pl.col(c.summary_excess_return))
    return AttributionResult(detail, summary, c)


def _linked(result, linking):
    if not callable(linking):
        raise TypeError("linking must be a callable linking function")
    linked = linking(result)
    if not isinstance(linked, AttributionResult) or linked.summary.height != 1:
        raise AttributionError("linking must return one-horizon AttributionResult")
    return linked


def _link_name(linking):
    for function, label in ((link_carino, "Cariño"), (link_grap, "GRAP"),
                            (link_frongello, "Frongello"), (link_menchero, "Menchero"),
                            (link_geometric, "Geometric")):
        if linking is function:
            return label
    if not callable(linking):
        raise TypeError("linking must be a callable linking function")
    return getattr(linking, "__name__", type(linking).__name__)


def _total(result, period, linking=link_menchero):
    c=result.columns
    if period is not None:
        rows = result.summary.filter(pl.col(c.period) == period)
        if rows.height != 1:
            raise AttributionError(f"period {period!r} is missing or duplicated")
        row=rows.row(0,named=True)
        return {"portfolio_return":row[c.summary_portfolio_return], "benchmark_return":row[c.summary_benchmark_return],
                "excess_return":row[c.summary_excess_return], "allocation":row[c.summary_allocation],
                "selection":row[c.summary_selection], "interaction":row[c.summary_interaction]}
    if result.summary.height == 1:
        return _total(result,result.summary[c.period][0])
    linked = _linked(result, linking)
    lc = linked.columns
    row = linked.summary.row(0,named=True)
    return {"portfolio_return":row[lc.summary_portfolio_return], "benchmark_return":row[lc.summary_benchmark_return],
            "excess_return":row[lc.summary_excess_return], "allocation":row[lc.summary_allocation],
            "selection":row[lc.summary_selection], "interaction":row[lc.summary_interaction]}


def _link_label(result, period, *, attributed, linking=link_menchero):
    if period is not None or result.summary.height < 2:
        return None
    return _link_name(linking) if attributed else "none"


def _chart_title(title, result, period, *, attributed, linking=link_menchero):
    label = _link_label(result, period, attributed=attributed, linking=linking)
    return title if label is None else f"{title} — Linking: {label}"


def _markdown(title, text, table, labels, *, period=None, linking_method=None):
    scope = f"Period {period}" if period is not None else "All periods"
    if linking_method is not None:
        scope += f" | Linking: {linking_method}"
    header = "| " + " | ".join(labels) + " |"
    separator = "| " + " | ".join(["---"] * len(labels)) + " |"
    lines = []
    for row in table.iter_rows():
        values = [f"{v * 100:.4f}%" if isinstance(v, float) else str(v).replace("|", "\\|").replace("\n", " ") for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return f"### {title} — {scope}\n\n{text}\n\n{header}\n{separator}\n" + "\n".join(lines)


def _display(text):
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
        if get_ipython() is not None:
            display(Markdown(text))
            return text
    except ImportError:
        pass
    print(text)
    return text


def _weights(result, period):
    c=result.columns
    detail = result.detail
    if period is not None:
        detail = detail.filter(pl.col(c.period) == period)
        if detail.is_empty():
            raise AttributionError(f"period {period!r} is not present")
        weights = detail.select(c.segment, pl.col(c.portfolio_weight).alias("_report_pw"), pl.col(c.benchmark_weight).alias("_report_bw"))
    else:
        periods = result.summary.height
        weights = detail.group_by(c.segment).agg(
            (pl.col(c.portfolio_weight).sum()/periods).alias("_report_pw"),
            (pl.col(c.benchmark_weight).sum()/periods).alias("_report_bw"))
    return weights.with_columns((pl.col("_report_pw")-pl.col("_report_bw")).alias("active_weight")).sort(c.segment)


def _returns(result, period):
    c=result.columns
    detail = result.detail
    if period is not None:
        detail = detail.filter(pl.col(c.period) == period)
        if detail.is_empty():
            raise AttributionError(f"period {period!r} is not present")
        return detail.select(c.segment,pl.col(c.portfolio_return).alias("_report_pr"),pl.col(c.benchmark_return).alias("_report_br")).sort(c.segment)
    if detail.group_by(c.segment).agg(pl.col(c.period).n_unique().alias("n")).filter(pl.col("n") != result.summary.height).height:
        raise AttributionError("aggregate category returns require each category in every period")
    return detail.group_by(c.segment).agg(
        ((1+pl.col(c.portfolio_return)).product()-1).alias("_report_pr"),
        ((1+pl.col(c.benchmark_return)).product()-1).alias("_report_br")).sort(c.segment)


def _attribution(result, period, linking=link_menchero):
    c=result.columns
    if period is not None:
        detail = result.detail.filter(pl.col(c.period) == period)
        if detail.is_empty():
            raise AttributionError(f"period {period!r} is not present")
        effects = detail.select(c.segment,pl.col(c.allocation).alias("allocation"),pl.col(c.selection).alias("selection"),pl.col(c.interaction).alias("interaction"))
    elif result.summary.height == 1:
        return _attribution(result,result.summary[c.period][0],linking)
    else:
        detail = _linked(result, linking).detail
        if c.segment not in detail.columns or not {"linked_allocation", "linked_selection", "linked_interaction"}.issubset(detail.columns):
            raise AttributionError("category attribution requires a linker with segment-level arithmetic effects")
        effects = detail.group_by(c.segment).agg(
            pl.col("linked_allocation").sum().alias("allocation"),
            pl.col("linked_selection").sum().alias("selection"),
            pl.col("linked_interaction").sum().alias("interaction"))
    return effects.with_columns(pl.sum_horizontal("allocation", "selection", "interaction").alias("active_effect")).sort(c.segment)


def _chart_table(table, columns, title, xlabel, *, segment="segment", stacked=False):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise ImportError("Chart functions require matplotlib; install brinsons[plot]") from exc
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.48*table.height+1.5)), layout="constrained")
    y = list(range(table.height))
    colors = ["#2366a8", "#9cabbc", "#e5a24f"] if not stacked else ["#2366a8", "#3f8d70", "#df9d50"]
    if stacked:
        positive, negative = [0.0]*len(y), [0.0]*len(y)
        for (key, label), color in zip(columns, colors):
            vals = table[key].to_list()
            left = [positive[i] if v >= 0 else negative[i] for i, v in enumerate(vals)]
            ax.barh(y, vals, left=left, color=color, label=label, height=.65)
            for i, value in enumerate(vals):
                if value >= 0: positive[i] += value
                else: negative[i] += value
    else:
        height = .72 / len(columns)
        for j, ((key, label), color) in enumerate(zip(columns, colors)):
            ax.barh([i + (j-(len(columns)-1)/2)*height for i in y], table[key].to_list(),
                    height=height, color=color, label=label)
    ax.set_yticks(y, table[segment].cast(pl.String).to_list())
    ax.invert_yaxis(); ax.axvline(0, color="#555555", linewidth=.7)
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title); ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True); ax.legend()
    return fig, ax


def print_category_weights(result: AttributionResult, period=None, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print Markdown category weights and return the Markdown string."""
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    table = _weights(result, period)
    note = "Active weight = portfolio weight − benchmark weight. Across all periods, weights are simple period averages."
    return _display(_markdown("Category Weights", note, table, ["Category", "Portfolio", "Benchmark", "Active"],
        period=period, linking_method=_link_label(result, period, attributed=False)))


def print_category_returns(result: AttributionResult, period=None, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print Markdown category returns and return the Markdown string."""
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    table = _returns(result, period)
    note = "Across all periods, category returns compound period category returns; within-category weights reset each period."
    return _display(_markdown("Category Returns", note, table, ["Category", "Portfolio", "Benchmark"],
        period=period, linking_method=_link_label(result, period, attributed=False)))


def print_category_attribution(result: AttributionResult, period=None, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Print Markdown category effects and return the Markdown string."""
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    table = _attribution(result, period, linking)
    note = "Allocation + selection + interaction = active contribution."
    if period is None and result.summary.height > 1:
        note += f" All-period effects use {_link_name(linking)} linking."
    return _display(_markdown("Category Attribution", note, table, ["Category", "Allocation", "Selection", "Interaction", "Active"],
        period=period, linking_method=_link_label(result, period, attributed=True, linking=linking)))


def print_total_attribution(result: AttributionResult, period=None, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Print Markdown total effects and return the Markdown string."""
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    totals = _total(result, period, linking)
    multi = period is None and result.summary.height > 1
    geometric = multi and linking is link_geometric
    if geometric:
        table = pl.DataFrame([{"allocation":totals["allocation"], "selection":totals["selection"],
            "cross_term":totals["allocation"]*totals["selection"], "excess_return":totals["excess_return"]}])
        note = "Geometric effects reconcile multiplicatively: (1 + allocation)(1 + selection) - 1 = relative active return."
        labels = ["Allocation", "Selection incl. interaction", "Cross term", "Relative active return"]
    else:
        table = pl.DataFrame([{key: totals[key] for key in ("allocation", "selection", "interaction", "excess_return")}])
        note = (f"All-period effects use {_link_name(linking)} linking and reconcile to the difference of compounded returns."
                if multi else "Effects reconcile to the period's active return.")
        labels = ["Allocation", "Selection", "Interaction", "Active return"]
    return _display(_markdown("Total Attribution", note, table, labels,
        period=period, linking_method=_link_label(result, period, attributed=True, linking=linking)))


def print_attribution_summary(result: AttributionResult, period=None, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Print a Markdown performance-attribution summary."""
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    totals = _total(result, period, linking)
    geometric = period is None and result.summary.height > 1 and linking is link_geometric
    rows = [("Categories", str(result.detail[result.columns.segment].n_unique())),
            ("Periods", "1" if period is not None else str(result.summary.height))]
    rows += [(label, f"{totals[key]*100:.4f}%") for label, key in [
        ("Portfolio return", "portfolio_return"), ("Benchmark return", "benchmark_return"),
        (("Relative active return" if geometric else "Active return"), "excess_return"),
        ("Allocation", "allocation"), ("Selection", "selection"), ("Interaction", "interaction")]]
    if geometric:
        rows.append(("Allocation × selection", f"{totals['allocation']*totals['selection']*100:.4f}%"))
    table = pl.DataFrame(rows, schema=["Measure", "Value"], orient="row")
    note = (f"All-period effects use {_link_name(linking)} linking."
            if period is None and result.summary.height > 1 else "Effects describe the selected period.")
    return _display(_markdown("Summary for Performance Attribution", note, table, ["Measure", "Value"],
        period=period, linking_method=_link_label(result, period, attributed=True, linking=linking)))


def chart_category_returns(result: AttributionResult, period=None, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Create horizontal category return bars; return (figure, axes)."""
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    label = "period returns" if period is not None else "compounded category returns"
    return _chart_table(_returns(result, period), [("_report_pr", "Portfolio"), ("_report_br", "Benchmark")],
        _chart_title(f"Portfolio and benchmark {label}", result, period, attributed=False),
        "Return", segment=result.columns.segment)


def chart_category_weights(result: AttributionResult, period=None, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Create horizontal category weight bars; return (figure, axes)."""
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    label = "period category weights" if period is not None else "average category weights"
    return _chart_table(_weights(result, period), [("_report_pw", "Portfolio"), ("_report_bw", "Benchmark"), ("active_weight", "Active")],
        _chart_title(f"Portfolio, benchmark and active {label}", result, period, attributed=False),
        "Weight", segment=result.columns.segment)


def chart_category_attribution(result: AttributionResult, period=None, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Create stacked horizontal attribution bars; return (figure, axes)."""
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    label = "period effects" if period is not None else "full-horizon effects"
    return _chart_table(_attribution(result, period, linking), [("allocation", "Allocation"), ("selection", "Selection"), ("interaction", "Interaction")],
        _chart_title(f"Category attribution — {label}", result, period, attributed=True, linking=linking),
        "Active contribution", segment=result.columns.segment, stacked=True)


def _require_multiple_periods(result):
    if not isinstance(result, AttributionResult):
        raise TypeError("expected an AttributionResult")
    if result.summary.height < 2:
        raise AttributionError("multiple-period reporting requires at least two periods")


def _require_single_period(period):
    if period is None:
        raise AttributionError("single-period reporting requires a period")



def print_single_period_category_weights(result: AttributionResult, period, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print category weights for single period."""
    _require_single_period(period)
    return print_category_weights(result, period=period, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def print_multiple_period_category_weights(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print category weights for multiple periods."""
    _require_multiple_periods(result)
    return print_category_weights(result, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def print_single_period_category_returns(result: AttributionResult, period, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print category returns for single period."""
    _require_single_period(period)
    return print_category_returns(result, period=period, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def print_multiple_period_category_returns(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print category returns for multiple periods."""
    _require_multiple_periods(result)
    return print_category_returns(result, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def print_single_period_category_attribution(result: AttributionResult, period, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print category attribution for single period."""
    _require_single_period(period)
    return print_category_attribution(result, period=period, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def print_multiple_period_category_attribution(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Print category attribution for multiple periods."""
    _require_multiple_periods(result)
    return print_category_attribution(result, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction, linking=linking)


def print_single_period_total_attribution(result: AttributionResult, period, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print total attribution for single period."""
    _require_single_period(period)
    return print_total_attribution(result, period=period, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def print_multiple_period_total_attribution(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Print total attribution for multiple periods."""
    _require_multiple_periods(result)
    return print_total_attribution(result, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction, linking=linking)


def print_single_period_attribution_summary(result: AttributionResult, period, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Print attribution summary for single period."""
    _require_single_period(period)
    return print_attribution_summary(result, period=period, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def print_multiple_period_attribution_summary(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Print attribution summary for multiple periods."""
    _require_multiple_periods(result)
    return print_attribution_summary(result, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction, linking=linking)


def chart_single_period_category_returns(result: AttributionResult, period, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Chart category returns for single period."""
    _require_single_period(period)
    return chart_category_returns(result, period=period, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def chart_multiple_period_category_returns(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Chart category returns for multiple periods."""
    _require_multiple_periods(result)
    return chart_category_returns(result, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def chart_single_period_category_weights(result: AttributionResult, period, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Chart category weights for single period."""
    _require_single_period(period)
    return chart_category_weights(result, period=period, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def chart_multiple_period_category_weights(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Chart category weights for multiple periods."""
    _require_multiple_periods(result)
    return chart_category_weights(result, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def chart_single_period_category_attribution(result: AttributionResult, period, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Chart category attribution for single period."""
    _require_single_period(period)
    return chart_category_attribution(result, period=period, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction)


def chart_multiple_period_category_attribution(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Chart category attribution for multiple periods."""
    _require_multiple_periods(result)
    return chart_category_attribution(result, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return,
        column_excess_return=column_excess_return, column_allocation=column_allocation,
        column_selection=column_selection, column_interaction=column_interaction, linking=linking)


def _waterfall(result: AttributionResult, period, by_category: bool, linking=link_menchero):
    """Plot signed effects, including the geometric cross term when selected."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Chart functions require matplotlib; install brinsons[plot]") from exc
    geometric = period is None and linking is link_geometric
    if geometric:
        if by_category:
            raise AttributionError("geometric linking has no category-level linked effects; use by_category=False")
        totals = _total(result, period, linking)
        allocation, selection = totals["allocation"], totals["selection"]
        labels = ["Allocation", "Selection incl. interaction", "Allocation × selection"]
        values = [allocation * 10_000, selection * 10_000, allocation * selection * 10_000]
        target_label = "Relative active return"
    else:
        effects = _attribution(result, period, linking)
        totals = _total(result, period, linking)
        categories = effects[result.columns.segment].cast(pl.String).to_list()
        labels, values = [], []
        for name, label in (("allocation", "Allocation"), ("selection", "Selection"), ("interaction", "Interaction")):
            if by_category:
                labels.extend(f"{segment}\n{label}" for segment in categories)
                values.extend(v * 10_000 for v in effects[name].to_list())
            else:
                labels.append(label)
                values.append(effects[name].sum() * 10_000)
        target_label = "Total active return"
    target = totals["excess_return"] * 10_000
    if abs(sum(values) - target) > 1e-5:
        raise AttributionError("waterfall effects do not reconcile to active return")
    count = len(values)
    fig, ax = plt.subplots(figsize=(max(9, min(30, count * .72 + 2)), 6.2), layout="constrained")
    cumulative = 0.0
    levels = [0.0]
    for index, value in enumerate(values):
        previous = cumulative
        cumulative += value
        levels.append(cumulative)
        color = "#2877b9" if value > 0 else "#c73c32" if value < 0 else "#b7bec5"
        ax.bar(index, abs(value), bottom=min(previous, cumulative), color=color, width=.68)
        ax.plot([index + .34, index + .66], [cumulative, cumulative], color="#999999", linewidth=.7)
        ax.annotate("0.00" if abs(value) < .005 else f"{value:+.2f}", (index, cumulative),
            xytext=(0, 4 if value >= 0 else -5), textcoords="offset points",
            ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax.bar(count, abs(target), bottom=min(0.0, target), color="#555555", width=.68)
    ax.annotate("0.00" if abs(target) < .005 else f"{target:+.2f}", (count, target),
        xytext=(0, 4 if target >= 0 else -5), textcoords="offset points",
        ha="center", va="bottom" if target >= 0 else "top", fontsize=8, fontweight="bold")
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set_xticks(list(range(count+1)), labels+[target_label], rotation=45, ha="right")
    ax.set_ylabel("Relative active return contribution (basis points)" if geometric
                  else "Active return contribution (basis points)")
    scope = f"period {period}" if period is not None else f"multiple periods | Linking: {_link_name(linking)}"
    ax.set_title(f"Excess-return attribution waterfall — {scope}", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e6e9ed", linewidth=.6)
    ax.set_axisbelow(True)
    low, high = min(*levels, target), max(*levels, target)
    padding = max((high-low)*.15, .5)
    ax.set_ylim(low-padding, high+padding)
    return fig, ax


def chart_single_period_excess_return_waterfall(result: AttributionResult, period, *, by_category: bool = True,
        column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction"):
    """Show one period's category effects and active return in basis points.

    Set ``by_category=False`` to show just allocation, selection, interaction,
    and total. Returns a Matplotlib ``(figure, axes)`` pair.
    """
    _require_single_period(period)
    if not isinstance(by_category, bool):
        raise TypeError("by_category must be a bool")
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    return _waterfall(result, period, by_category)


def chart_multiple_period_excess_return_waterfall(result: AttributionResult, *, by_category: bool | None = None,
        column_period: str = "period", column_segment: str = "segment",
        column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation",
        column_selection: str = "selection", column_interaction: str = "interaction",
        linking: Callable = link_menchero):
    """Show a chosen linker's effects and active return in basis points.

    Geometric linking uses aggregate allocation, selection, and their cross term.
    Its category-level waterfall is unavailable because that linker does not
    produce linked effects by category.
    """
    _require_multiple_periods(result)
    if by_category is None:
        by_category = linking is not link_geometric
    if not isinstance(by_category, bool):
        raise TypeError("by_category must be a bool")
    result = _inputs(result, column_period, column_segment, column_portfolio_weight, column_benchmark_weight,
        column_portfolio_return, column_benchmark_return, column_excess_return, column_allocation, column_selection, column_interaction)
    return _waterfall(result, None, by_category, linking)
