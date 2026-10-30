"""Three explicit, Polars-based multi-account attribution workflows.

Amounts are beginning-of-period market values or benchmark notionals. The
module leaves the existing single-account calculation interfaces untouched.
"""
from dataclasses import dataclass
from typing import Callable

import polars as pl

from .columns import ColumnNames
from .core import AttributionError, AttributionResult, _check, _require, fachler, link_menchero


@dataclass(frozen=True)
class MultiAccountColumns:
    """Input roles retained by multi-account results; names are never renamed."""

    period: str = "period"
    account: str = "account"
    sector: str = "sector"
    security: str = "security"
    portfolio_value: str = "portfolio_value"
    benchmark_value: str = "benchmark_value"
    portfolio_return: str = "portfolio_return"
    benchmark_return: str = "benchmark_return"


@dataclass(frozen=True)
class MacroMicroAttribution:
    """Account Brinson, within-account Brinson, and reconciled replacement."""

    macro: AttributionResult
    micro_detail: pl.DataFrame
    micro_summary: pl.DataFrame
    integrated_detail: pl.DataFrame
    integrated_summary: pl.DataFrame
    columns: MultiAccountColumns


@dataclass(frozen=True)
class HierarchicalDecisionAttribution:
    """Decision effects and successive benchmark-to-portfolio notionals."""

    detail: pl.DataFrame
    notionals: pl.DataFrame
    summary: pl.DataFrame
    columns: MultiAccountColumns


@dataclass(frozen=True)
class LinkedMultiAccountAttribution:
    """Arithmetic full-horizon detail and summary with named linking method."""

    detail: pl.DataFrame
    summary: pl.DataFrame
    linking_method: str


def _schema(period, account, sector, security, portfolio_value, benchmark_value,
            portfolio_return, benchmark_return) -> MultiAccountColumns:
    c = MultiAccountColumns(period, account, sector, security, portfolio_value,
                            benchmark_value, portfolio_return, benchmark_return)
    identifiers = [c.period, c.account, c.sector, c.security]
    values = [c.portfolio_value, c.benchmark_value, c.portfolio_return, c.benchmark_return]
    if any(not isinstance(x, str) or not x for x in identifiers + values):
        raise AttributionError("column names must be nonempty strings")
    if len(set(identifiers)) != 4 or set(identifiers) & set(values):
        raise AttributionError("identifier columns must be distinct from each other and numeric columns")
    if c.portfolio_value == c.benchmark_value:
        raise AttributionError("portfolio and benchmark value columns must be distinct")
    return c


def _positions(data: pl.DataFrame, c: MultiAccountColumns, extra: str | None = None) -> pl.DataFrame:
    keys = [c.period, c.account, c.sector, c.security]
    nums = list(dict.fromkeys([c.portfolio_value, c.benchmark_value,
                               c.portfolio_return, c.benchmark_return]))
    required = keys + nums + ([extra] if extra is not None and extra not in keys else [])
    _require(data, required)
    if extra is not None and extra in nums:
        raise AttributionError("micro segment must identify categories, not numeric values")
    if data.select(pl.struct(keys).is_duplicated().any()).item():
        raise AttributionError("duplicate period/account/sector/security")
    try:
        selected = data.select(required).with_columns(pl.col(nums).cast(pl.Float64))
        if selected.select(pl.any_horizontal(~pl.col(n).is_finite() for n in nums).any()).item():
            raise AttributionError("values and returns must be finite")
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise AttributionError("values and returns must be numeric and finite") from exc
    pv, bv = c.portfolio_value, c.benchmark_value
    if selected.select(((pl.col(pv) < 0) | (pl.col(bv) < 0)).any()).item():
        raise AttributionError("market values and benchmark notionals must be nonnegative")
    if selected.select(((pl.col(c.portfolio_return) < -1) |
                        (pl.col(c.benchmark_return) < -1)).any()).item():
        raise AttributionError("simple returns cannot be below -100%")
    account_totals = selected.group_by(c.period, c.account).agg(pl.col(pv, bv).sum())
    if account_totals.select(((pl.col(pv) <= 0) | (pl.col(bv) <= 0)).any()).item():
        raise AttributionError("every period/account needs positive portfolio and benchmark notional")
    return selected


def _accounts(d: pl.DataFrame, c: MultiAccountColumns) -> pl.DataFrame:
    pv, bv, rp, rb = (pl.col(x) for x in
                      (c.portfolio_value, c.benchmark_value, c.portfolio_return, c.benchmark_return))
    return d.group_by(c.period, c.account).agg(
        pv.sum().alias("_ma_pv"), bv.sum().alias("_ma_bv"),
        (pv * rp).sum().alias("_ma_pc"), (bv * rb).sum().alias("_ma_bc"),
    ).with_columns(
        (pl.col("_ma_pc") / pl.col("_ma_pv")).alias("_ma_rp"),
        (pl.col("_ma_bc") / pl.col("_ma_bv")).alias("_ma_rb"),
        (pl.col("_ma_pv") / pl.col("_ma_pv").sum().over(c.period)).alias("_ma_wp"),
        (pl.col("_ma_bv") / pl.col("_ma_bv").sum().over(c.period)).alias("_ma_wb"),
    ).sort(c.period, c.account)


def _run_account_method(data: pl.DataFrame, method: Callable, effects: int, *,
                        period: str, segment: str, weight_p: str, weight_b: str,
                        return_p: str, return_b: str) -> AttributionResult:
    if not callable(method):
        raise TypeError("attribution method must be a callable such as bhb or fachler")
    return method(data, effects=effects, column_period=period, column_segment=segment,
                  column_portfolio_weight=weight_p, column_benchmark_weight=weight_b,
                  column_portfolio_return=return_p, column_benchmark_return=return_b)


def macro_micro_attribution(positions: pl.DataFrame, *,
        macro_attribution: Callable = fachler, micro_attribution: Callable = fachler,
        macro_effects: int = 3, micro_effects: int = 3,
        column_period: str = "period", column_account: str = "account",
        column_sector: str = "sector", column_security: str = "security",
        column_micro_segment: str | None = None,
        column_portfolio_value: str = "portfolio_value",
        column_benchmark_value: str = "benchmark_value",
        column_portfolio_return: str = "portfolio_return",
        column_benchmark_return: str = "benchmark_return") -> MacroMicroAttribution:
    """Account-level Brinson followed by sector/security Brinson in each account.

    The integrated decomposition replaces macro selection plus interaction by
    portfolio-account-weighted micro effects, avoiding double counting.
    ``column_micro_segment`` defaults to ``column_sector``; it may name the
    security column or another categorical field in ``positions``.
    """
    c = _schema(column_period, column_account, column_sector, column_security,
                column_portfolio_value, column_benchmark_value,
                column_portfolio_return, column_benchmark_return)
    segment = c.sector if column_micro_segment is None else column_micro_segment
    if segment in (c.period, c.account):
        raise AttributionError("micro segment must differ from period and account")
    d = _positions(positions, c, segment)
    accounts = _accounts(d, c)
    macro = _run_account_method(accounts, macro_attribution, macro_effects,
                                period=c.period, segment=c.account, weight_p="_ma_wp",
                                weight_b="_ma_wb", return_p="_ma_rp", return_b="_ma_rb")
    pv, bv, rp, rb = (pl.col(x) for x in
                      (c.portfolio_value, c.benchmark_value, c.portfolio_return, c.benchmark_return))
    micro = d.group_by(c.period, c.account, segment).agg(
        pv.sum().alias("_ma_seg_pv"), bv.sum().alias("_ma_seg_bv"),
        (pv * rp).sum().alias("_ma_seg_pc"), (bv * rb).sum().alias("_ma_seg_bc"),
    )
    if micro.select(((pl.col("_ma_seg_pv") <= 0) | (pl.col("_ma_seg_bv") <= 0)).any()).item():
        raise AttributionError("every micro segment needs positive values on both sides")
    micro = micro.join(accounts.select(c.period, c.account, "_ma_pv", "_ma_bv", "_ma_wp"),
                       on=[c.period, c.account]).with_columns(
        (pl.col("_ma_seg_pv") / pl.col("_ma_pv")).alias("_ma_seg_wp"),
        (pl.col("_ma_seg_bv") / pl.col("_ma_bv")).alias("_ma_seg_wb"),
        (pl.col("_ma_seg_pc") / pl.col("_ma_seg_pv")).alias("_ma_seg_rp"),
        (pl.col("_ma_seg_bc") / pl.col("_ma_seg_bv")).alias("_ma_seg_rb"),
    )
    details = []
    summaries = []
    for account in accounts[c.account].unique().sort().to_list():
        one = _run_account_method(micro.filter(pl.col(c.account) == account),
                                  micro_attribution, micro_effects, period=c.period,
                                  segment=segment, weight_p="_ma_seg_wp", weight_b="_ma_seg_wb",
                                  return_p="_ma_seg_rp", return_b="_ma_seg_rb")
        details.append(one.detail)
        summaries.append(one.summary.with_columns(pl.lit(account).alias(c.account)))
    micro_detail = pl.concat(details).sort(c.period, c.account, segment).with_columns(
        (pl.col("_ma_wp") * pl.col(effect)).alias("weighted_micro_" + effect)
        for effect in ("allocation", "selection", "interaction"))
    micro_summary = pl.concat(summaries).join(
        accounts.select(c.period, c.account, pl.col("_ma_wp").alias("account_portfolio_weight")),
        on=[c.period, c.account]).sort(c.period, c.account)
    _check(micro_summary.join(accounts.select(c.period, c.account, "_ma_rp", "_ma_rb"),
                              on=[c.period, c.account]),
           pl.col("portfolio_return") - pl.col("benchmark_return") -
           (pl.col("_ma_rp") - pl.col("_ma_rb")))
    integrated_detail = macro.detail.select(
        c.period, c.account, pl.col("_ma_wp").alias("account_portfolio_weight"),
        pl.col("_ma_wb").alias("account_benchmark_weight"),
        pl.col("_ma_rp").alias("account_portfolio_return"),
        pl.col("_ma_rb").alias("account_benchmark_return"),
        pl.col("allocation").alias("account_allocation"),
        pl.col("selection").alias("macro_selection"),
        pl.col("interaction").alias("macro_interaction"),
    ).join(micro_summary.select(c.period, c.account,
        *(pl.col(x).alias("micro_" + x) for x in ("allocation", "selection", "interaction"))),
        on=[c.period, c.account]).with_columns(
        (pl.col("account_portfolio_weight") * pl.col("micro_" + x)).alias("weighted_micro_" + x)
        for x in ("allocation", "selection", "interaction"))
    _check(integrated_detail,
           pl.col("macro_selection") + pl.col("macro_interaction") -
           pl.sum_horizontal("weighted_micro_allocation", "weighted_micro_selection",
                             "weighted_micro_interaction"))
    integrated_detail = integrated_detail.with_columns(pl.sum_horizontal(
        "account_allocation", "weighted_micro_allocation", "weighted_micro_selection",
        "weighted_micro_interaction").alias("total_effect")).sort(c.period, c.account)
    integrated_summary = integrated_detail.group_by(c.period).agg(
        pl.col("account_allocation", "weighted_micro_allocation", "weighted_micro_selection",
               "weighted_micro_interaction", "total_effect").sum(),
    ).join(macro.summary.select(c.period, "portfolio_return", "benchmark_return", "excess_return"),
           on=c.period).sort(c.period)
    _check(integrated_summary, pl.col("total_effect") - pl.col("excess_return"))
    return MacroMicroAttribution(macro, micro_detail, micro_summary,
                                 integrated_detail, integrated_summary, c)


def value_weighted_total_attribution(positions: pl.DataFrame, *,
        attribution: Callable = fachler, effects: int = 3,
        column_period: str = "period", column_account: str = "account",
        column_sector: str = "sector", column_security: str = "security",
        column_segment: str | None = None,
        column_portfolio_value: str = "portfolio_value",
        column_benchmark_value: str = "benchmark_value",
        column_portfolio_return: str = "portfolio_return",
        column_benchmark_return: str = "benchmark_return") -> AttributionResult:
    """Pool market values and benchmark notionals before one Brinson calculation.

    ``column_segment`` defaults to ``column_sector``; using a shared security
    identifier also works. Account configuration is intentionally not reported.
    """
    c = _schema(column_period, column_account, column_sector, column_security,
                column_portfolio_value, column_benchmark_value,
                column_portfolio_return, column_benchmark_return)
    segment = c.sector if column_segment is None else column_segment
    if segment in (c.period, c.account):
        raise AttributionError("pooled segment must differ from period and account")
    d = _positions(positions, c, segment)
    pv, bv, rp, rb = (pl.col(x) for x in
                      (c.portfolio_value, c.benchmark_value, c.portfolio_return, c.benchmark_return))
    pooled = d.group_by(c.period, segment).agg(
        pv.sum().alias("_ma_seg_pv"), bv.sum().alias("_ma_seg_bv"),
        (pv * rp).sum().alias("_ma_seg_pc"), (bv * rb).sum().alias("_ma_seg_bc"),
    )
    if pooled.select(((pl.col("_ma_seg_pv") <= 0) | (pl.col("_ma_seg_bv") <= 0)).any()).item():
        raise AttributionError("every pooled segment needs positive values on both sides")
    pooled = pooled.with_columns(
        (pl.col("_ma_seg_pv") / pl.col("_ma_seg_pv").sum().over(c.period)).alias("_ma_total_wp"),
        (pl.col("_ma_seg_bv") / pl.col("_ma_seg_bv").sum().over(c.period)).alias("_ma_total_wb"),
        (pl.col("_ma_seg_pc") / pl.col("_ma_seg_pv")).alias("_ma_total_rp"),
        (pl.col("_ma_seg_bc") / pl.col("_ma_seg_bv")).alias("_ma_total_rb"),
    )
    return _run_account_method(pooled, attribution, effects, period=c.period, segment=segment,
                               weight_p="_ma_total_wp", weight_b="_ma_total_wb",
                               return_p="_ma_total_rp", return_b="_ma_total_rb")


def hierarchical_decision_attribution(positions: pl.DataFrame, *,
        column_period: str = "period", column_account: str = "account",
        column_sector: str = "sector", column_security: str = "security",
        column_portfolio_value: str = "portfolio_value",
        column_benchmark_value: str = "benchmark_value",
        column_portfolio_return: str = "portfolio_return",
        column_benchmark_return: str = "benchmark_return") -> HierarchicalDecisionAttribution:
    """Successive notionals: benchmark -> account -> sector -> stock -> return.

    BF-centered allocation is used at each parent. The four stage differences
    exactly reconcile to the observed total portfolio-minus-benchmark return.
    Account/sector benchmark conditional weights must be defined on both sides.
    """
    c = _schema(column_period, column_account, column_sector, column_security,
                column_portfolio_value, column_benchmark_value,
                column_portfolio_return, column_benchmark_return)
    d = _positions(positions, c)
    accounts = _accounts(d, c)
    pv, bv, rp, rb = (pl.col(x) for x in
                      (c.portfolio_value, c.benchmark_value, c.portfolio_return, c.benchmark_return))
    sectors = d.group_by(c.period, c.account, c.sector).agg(
        pv.sum().alias("_ma_sector_pv"), bv.sum().alias("_ma_sector_bv"),
        (bv * rb).sum().alias("_ma_sector_bc"),
    )
    if sectors.select(((pl.col("_ma_sector_pv") <= 0) |
                       (pl.col("_ma_sector_bv") <= 0)).any()).item():
        raise AttributionError("every account/sector needs positive values on both sides")
    sectors = sectors.join(accounts.select(c.period, c.account, "_ma_pv", "_ma_bv", "_ma_wp", "_ma_rb"),
                           on=[c.period, c.account]).with_columns(
        (pl.col("_ma_sector_pv") / pl.col("_ma_pv")).alias("_ma_sector_wp"),
        (pl.col("_ma_sector_bv") / pl.col("_ma_bv")).alias("_ma_sector_wb"),
        (pl.col("_ma_sector_bc") / pl.col("_ma_sector_bv")).alias("_ma_sector_rb"),
    )
    stocks = d.join(sectors.select(c.period, c.account, c.sector, "_ma_wp",
                                    "_ma_sector_wp", "_ma_sector_pv", "_ma_sector_bv",
                                    "_ma_sector_rb"),
                    on=[c.period, c.account, c.sector]).with_columns(
        (pv / pl.col("_ma_sector_pv")).alias("_ma_stock_wp"),
        (bv / pl.col("_ma_sector_bv")).alias("_ma_stock_wb"),
    )
    root = accounts.group_by(c.period).agg(
        (pl.col("_ma_wp") * pl.col("_ma_rp")).sum().alias("_n4"),
        (pl.col("_ma_wb") * pl.col("_ma_rb")).sum().alias("_n0"),
        (pl.col("_ma_wp") * pl.col("_ma_rb")).sum().alias("_n1"),
    )
    sector_notional = sectors.group_by(c.period).agg(
        (pl.col("_ma_wp") * pl.col("_ma_sector_wp") * pl.col("_ma_sector_rb"))
        .sum().alias("_n2"))
    stock_notional = stocks.group_by(c.period).agg(
        (pl.col("_ma_wp") * pl.col("_ma_sector_wp") * pl.col("_ma_stock_wp") *
         pl.col(c.benchmark_return)).sum().alias("_n3"))
    wide = root.join(sector_notional, on=c.period).join(stock_notional, on=c.period).sort(c.period)
    account_rows = accounts.join(root.select(c.period, "_n0"), on=c.period).select(
        c.period, c.account,
        pl.lit(None, dtype=d.schema[c.sector]).alias(c.sector),
        pl.lit(None, dtype=d.schema[c.security]).alias(c.security),
        pl.lit("account_allocation").alias("stage"),
        ((pl.col("_ma_wp") - pl.col("_ma_wb")) *
         (pl.col("_ma_rb") - pl.col("_n0"))).alias("effect"),
    )
    sector_rows = sectors.select(
        c.period, c.account, c.sector,
        pl.lit(None, dtype=d.schema[c.security]).alias(c.security),
        pl.lit("sector_allocation").alias("stage"),
        (pl.col("_ma_wp") * (pl.col("_ma_sector_wp") - pl.col("_ma_sector_wb")) *
         (pl.col("_ma_sector_rb") - pl.col("_ma_rb"))).alias("effect"),
    )
    stock_allocation_rows = stocks.select(
        c.period, c.account, c.sector, c.security,
        pl.lit("security_allocation").alias("stage"),
        (pl.col("_ma_wp") * pl.col("_ma_sector_wp") *
         (pl.col("_ma_stock_wp") - pl.col("_ma_stock_wb")) *
         (pl.col(c.benchmark_return) - pl.col("_ma_sector_rb"))).alias("effect"),
    )
    stock_selection_rows = stocks.select(
        c.period, c.account, c.sector, c.security,
        pl.lit("security_selection").alias("stage"),
        (pl.col("_ma_wp") * pl.col("_ma_sector_wp") * pl.col("_ma_stock_wp") *
         (pl.col(c.portfolio_return) - pl.col(c.benchmark_return))).alias("effect"),
    )
    stage_names = ("account_allocation", "sector_allocation",
                   "security_allocation", "security_selection")
    detail = pl.concat([account_rows, sector_rows, stock_allocation_rows,
                        stock_selection_rows]).sort(c.period, "stage", c.account, c.sector, c.security)
    effects = detail.group_by(c.period).agg(
        *(pl.col("effect").filter(pl.col("stage") == stage).sum().alias(stage)
          for stage in stage_names))
    summary = wide.join(effects, on=c.period).with_columns(
        pl.col("_n4").alias("portfolio_return"),
        pl.col("_n0").alias("benchmark_return"),
        (pl.col("_n4") - pl.col("_n0")).alias("excess_return"),
        pl.sum_horizontal(*stage_names).alias("total_effect"),
    ).sort(c.period)
    for before, after, stage in zip(("_n0", "_n1", "_n2", "_n3"),
                                     ("_n1", "_n2", "_n3", "_n4"), stage_names):
        _check(summary, pl.col(after) - pl.col(before) - pl.col(stage))
    _check(summary, pl.col("total_effect") - pl.col("excess_return"))
    notionals = pl.concat([
        wide.select(c.period, pl.lit(index).alias("stage_index"),
                    pl.lit(name).alias("stage"), pl.col(field).alias("return"))
        for index, (name, field) in enumerate(zip(
            ("benchmark", "account_weights", "sector_weights",
             "security_weights", "portfolio_returns"),
            ("_n0", "_n1", "_n2", "_n3", "_n4")))
    ]).sort(c.period, "stage_index")
    return HierarchicalDecisionAttribution(
        detail, notionals,
        summary.select(c.period, *stage_names, "portfolio_return", "benchmark_return",
                       "excess_return", "total_effect"), c)


def _period_factors(base: AttributionResult, linking: Callable, period: str):
    if not callable(linking):
        raise TypeError("linking must be a callable such as link_menchero")
    linked = linking(base)
    if not isinstance(linked, AttributionResult) or "factor" not in linked.detail.columns:
        raise AttributionError("multi-account effect linking requires an arithmetic linker with period factors")
    factors = linked.detail.group_by(period).agg(pl.col("factor").first()).sort(period)
    if factors.height != base.summary.height:
        raise AttributionError("linking factors do not match all periods")
    return factors, linked.summary


def link_macro_micro_attribution(result: MacroMicroAttribution, *,
        linking: Callable = link_menchero) -> LinkedMultiAccountAttribution:
    """Link the integrated account-allocation and weighted micro effects.

    ``linking`` must be an arithmetic factor linker such as Cariño, GRAP,
    Frongello, or Menchero. The same period factor is applied to every
    account/effect channel, preserving the integrated identity.
    """
    if not isinstance(result, MacroMicroAttribution):
        raise TypeError("expected a MacroMicroAttribution")
    period = result.columns.period
    factors, total = _period_factors(result.macro, linking, period)
    effects = ("account_allocation", "weighted_micro_allocation",
               "weighted_micro_selection", "weighted_micro_interaction")
    detail = result.integrated_detail.join(factors, on=period).with_columns(
        (pl.col(name) * pl.col("factor")).alias("linked_" + name) for name in effects)
    summary = detail.select(*(pl.col("linked_" + name).sum().alias(name) for name in effects)).join(
        total.select("portfolio_return", "benchmark_return", "excess_return"), how="cross").with_columns(
        pl.sum_horizontal(*effects).alias("total_effect"))
    _check(summary, pl.col("total_effect") - pl.col("excess_return"))
    return LinkedMultiAccountAttribution(detail.sort(period, result.columns.account), summary,
                                         getattr(linking, "__name__", type(linking).__name__))


def link_hierarchical_decision_attribution(result: HierarchicalDecisionAttribution, *,
        linking: Callable = link_menchero) -> LinkedMultiAccountAttribution:
    """Link all four successive-notional stage effects with one arithmetic method."""
    if not isinstance(result, HierarchicalDecisionAttribution):
        raise TypeError("expected a HierarchicalDecisionAttribution")
    c = result.columns
    stages = ("account_allocation", "sector_allocation", "security_allocation",
              "security_selection")
    temporary_period, temporary_segment = "_ma_link_period", "_ma_link_segment"
    if temporary_period in result.summary.columns or temporary_segment in result.summary.columns:
        raise AttributionError("reserved linking field conflicts with result columns")
    period_summary = result.summary.select(
        pl.col(c.period).alias(temporary_period),
        pl.col("account_allocation").alias("allocation"),
        pl.sum_horizontal("sector_allocation", "security_allocation", "security_selection").alias("selection"),
        pl.lit(0.0).alias("interaction"),
        "portfolio_return", "benchmark_return", "excess_return")
    period_detail = period_summary.select(
        temporary_period, pl.lit("total").alias(temporary_segment),
        "allocation", "selection", "interaction")
    base = AttributionResult(period_detail, period_summary,
                             ColumnNames(period=temporary_period, segment=temporary_segment))
    factors, total = _period_factors(base, linking, temporary_period)
    factors = factors.select(pl.col(temporary_period).alias(c.period), "factor")
    detail = result.detail.join(factors, on=c.period).with_columns(
        (pl.col("effect") * pl.col("factor")).alias("linked_effect"))
    summary = detail.select(*(pl.col("linked_effect").filter(pl.col("stage") == stage)
                              .sum().alias(stage) for stage in stages)).join(
        total.select("portfolio_return", "benchmark_return", "excess_return"), how="cross").with_columns(
        pl.sum_horizontal(*stages).alias("total_effect"))
    _check(summary, pl.col("total_effect") - pl.col("excess_return"))
    return LinkedMultiAccountAttribution(detail.sort(c.period, "stage", c.account, c.sector, c.security),
                                         summary, getattr(linking, "__name__", type(linking).__name__))
