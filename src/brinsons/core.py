"""Financial attribution calculations; returns and weights are decimal values."""
from dataclasses import dataclass, field, replace
import polars as pl
from .columns import ColumnNames


class AttributionError(ValueError):
    """Invalid attribution inputs or failed reconciliation."""


@dataclass(frozen=True)
class AttributionResult:
    detail: pl.DataFrame
    summary: pl.DataFrame
    columns: ColumnNames = field(default_factory=ColumnNames)


_TOL = 1e-9


def _require(df, cols):
    if not isinstance(df, pl.DataFrame):
        raise TypeError("expected a Polars DataFrame")
    missing = set(cols) - set(df.columns)
    if missing:
        raise AttributionError(f"missing columns: {sorted(missing)}")
    if df.is_empty():
        raise AttributionError("input is empty")
    if df.select(pl.any_horizontal(pl.col(c).is_null() for c in cols).any()).item():
        raise AttributionError("required columns contain nulls")


def _validate(df, cols, columns: ColumnNames | None = None):
    columns = columns or ColumnNames()
    keys = list(columns.keys)
    weights = list(columns.weights)
    _require(df, keys + cols)
    if df.select(pl.struct(keys).is_duplicated().any()).item():
        raise AttributionError("duplicate period/segment")
    try:
        df = df.with_columns(pl.col(list(dict.fromkeys(cols))).cast(pl.Float64))
        if df.select(pl.any_horizontal(~pl.col(c).is_finite() for c in cols).any()).item():
            raise AttributionError("values must be finite")
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise AttributionError("numeric columns must contain finite numbers") from exc
    totals = df.group_by(columns.period).agg(pl.col(list(dict.fromkeys(weights))).sum())
    if totals.select(pl.any_horizontal((pl.col(c) - 1).abs() > 1e-8 for c in weights).any()).item():
        raise AttributionError("portfolio and benchmark weights must sum to one per period")
    return df


def _check(df, expression):
    if df.select((expression.abs() > _TOL).any()).item():
        raise AttributionError("attribution does not reconcile")


def _resolve_input_columns(columns, period, segment, portfolio_weight, benchmark_weight, portfolio_return,
                           benchmark_return):
    supplied = ColumnNames(period=period, segment=segment, portfolio_weight=portfolio_weight,
                           benchmark_weight=benchmark_weight, portfolio_return=portfolio_return,
                           benchmark_return=benchmark_return)
    if columns is not None:
        if not isinstance(columns, ColumnNames):
            raise TypeError("columns must be a ColumnNames instance")
        if supplied != ColumnNames():
            raise AttributionError("pass either columns or individual column arguments, not both")
        supplied = columns
    names = [supplied.period, supplied.segment, supplied.portfolio_weight, supplied.benchmark_weight,
             supplied.portfolio_return, supplied.benchmark_return]
    if any(not isinstance(x, str) or not x for x in names):
        raise AttributionError("column names must be nonempty strings")
    if supplied.period == supplied.segment or {supplied.period, supplied.segment} & set(names[2:]):
        raise AttributionError("period and segment must be distinct from each other and numeric input columns")
    return supplied


def bhb(data: pl.DataFrame, effects: int = 3, *, column_period: str = "period",
        column_segment: str = "segment", column_portfolio_weight: str = "portfolio_weight",
        column_benchmark_weight: str = "benchmark_weight", column_portfolio_return: str = "portfolio_return",
        column_benchmark_return: str = "benchmark_return", columns: ColumnNames | None = None) -> AttributionResult:
    """Calculate Brinson鈥揌ood鈥揃eebower attribution."""
    schema = _resolve_input_columns(columns, column_period, column_segment, column_portfolio_weight,
                                    column_benchmark_weight, column_portfolio_return, column_benchmark_return)
    return _attribute(data, pl.lit(0.0), effects, schema)


def fachler(data: pl.DataFrame, effects: int = 3, *, column_period: str = "period",
            column_segment: str = "segment", column_portfolio_weight: str = "portfolio_weight",
            column_benchmark_weight: str = "benchmark_weight", column_portfolio_return: str = "portfolio_return",
            column_benchmark_return: str = "benchmark_return", columns: ColumnNames | None = None) -> AttributionResult:
    """Calculate Brinson鈥揊achler attribution."""
    schema = _resolve_input_columns(columns, column_period, column_segment, column_portfolio_weight,
                                    column_benchmark_weight, column_portfolio_return, column_benchmark_return)
    benchmark = (pl.col(schema.benchmark_weight) * pl.col(schema.benchmark_return)).sum().over(schema.period)
    return _attribute(data, benchmark, effects, schema)


def _attribute(data: pl.DataFrame, benchmark: pl.Expr, effects: int, columns: ColumnNames) -> AttributionResult:
    """Calculate segment effects using the supplied allocation reference.

    Required columns: period, segment, portfolio_weight, benchmark_weight,
    portfolio_return, benchmark_return. Period must sort chronologically.
    """
    if effects not in (2, 3):
        raise AttributionError("effects must be 2 or 3")
    names = [columns.portfolio_weight, columns.benchmark_weight, columns.portfolio_return, columns.benchmark_return]
    d = _validate(data, names, columns)
    generated = [*columns.effects, "portfolio_contribution", "benchmark_contribution"]
    if len(generated) != len(set(generated)):
        raise AttributionError("calculated effect column names must be distinct")
    outputs = set(generated)
    if outputs & set(d.columns):
        raise AttributionError(f"input columns conflict with generated effects: {sorted(outputs & set(d.columns))}")
    summary_names = [columns.period, *columns.summary_effects, columns.summary_portfolio_return,
                     columns.summary_benchmark_return, columns.summary_excess_return]
    if len(summary_names) != len(set(summary_names)):
        raise AttributionError("calculated summary column names must be distinct")
    wp, wb, rp, rb = (pl.col(c) for c in names)
    active_w, active_r = wp - wb, rp - rb
    detail = d.with_columns(
        (active_w * (rb - benchmark)).alias(columns.allocation),
        ((wb if effects == 3 else wp) * active_r).alias(columns.selection),
        (active_w * active_r if effects == 3 else pl.lit(0.0)).alias(columns.interaction),
        (wp * rp).alias("portfolio_contribution"),
        (wb * rb).alias("benchmark_contribution"),
    ).sort([columns.period, columns.segment])
    summary = detail.group_by(columns.period).agg(
        *(pl.col(source).sum().alias(target) for source, target in zip(columns.effects, columns.summary_effects)),
        pl.col("portfolio_contribution").sum().alias(columns.summary_portfolio_return),
        pl.col("benchmark_contribution").sum().alias(columns.summary_benchmark_return),
    ).with_columns((pl.col(columns.summary_portfolio_return) - pl.col(columns.summary_benchmark_return)).alias(
        columns.summary_excess_return)).sort(columns.period)
    _check(summary, pl.sum_horizontal(*columns.summary_effects) - pl.col(columns.summary_excess_return))
    return AttributionResult(detail, summary, columns)


def currency_attribution(data: pl.DataFrame, *, column_period: str = "period", column_segment: str = "segment",
                         column_portfolio_weight: str = "portfolio_weight",
                         column_benchmark_weight: str = "benchmark_weight",
                         column_portfolio_return: str = "portfolio_return",
                         column_benchmark_return: str = "benchmark_return",
                         column_local_return: str = "local_return", column_currency_return: str = "currency_return",
                         columns: ColumnNames | None = None) -> AttributionResult:
    """Calculate pybrinson's additive local/currency split (not full hedged KS)."""
    schema = _resolve_input_columns(columns, column_period, column_segment, column_portfolio_weight,
                                    column_benchmark_weight, column_portfolio_return, column_benchmark_return)
    if columns is not None and (column_local_return != "local_return" or column_currency_return != "currency_return"):
        raise AttributionError("pass either columns or individual column arguments, not both")
    if columns is None:
        schema = replace(schema, local_return=column_local_return, currency_return=column_currency_return)
    names = [schema.portfolio_weight, schema.benchmark_weight, schema.portfolio_return,
             schema.benchmark_return, schema.local_return, schema.currency_return]
    d = _validate(data, names, schema)
    output_names = {"market_allocation", "security_selection", "currency_allocation", "interaction",
                    "portfolio_contribution", "benchmark_contribution"}
    if output_names & set(d.columns):
        raise AttributionError("input columns conflict with generated currency effects")
    _check(d, pl.col(schema.portfolio_return) - pl.col(schema.local_return) - pl.col(schema.currency_return))
    wp, wb = pl.col(schema.portfolio_weight), pl.col(schema.benchmark_weight)
    rp, rb = pl.col(schema.portfolio_return), pl.col(schema.benchmark_return)
    local, fx = pl.col(schema.local_return), pl.col(schema.currency_return)
    lb = rb - fx
    dw, dr = wp - wb, local - lb
    detail = d.with_columns(
        (dw * (lb - (wb * lb).sum().over(schema.period))).alias("market_allocation"),
        (wb * dr).alias("security_selection"),
        (dw * (fx - (wb * fx).sum().over(schema.period))).alias("currency_allocation"),
        (dw * dr).alias("interaction"),
        (wp * rp).alias("portfolio_contribution"), (wb * rb).alias("benchmark_contribution"),
    ).sort([schema.period, schema.segment])
    names = ["market_allocation", "security_selection", "currency_allocation", "interaction"]
    summary = detail.group_by(schema.period).agg(pl.col(names).sum(),
                                                 pl.col("portfolio_contribution").sum().alias("portfolio_return"),
                                                 pl.col("benchmark_contribution").sum().alias("benchmark_return"),
                                                 ).with_columns(
        (pl.col("portfolio_return") - pl.col("benchmark_return")).alias("excess_return")).sort(schema.period)
    _check(summary, pl.sum_horizontal(names) - pl.col("excess_return"))
    return AttributionResult(detail, summary, schema)


def prepare(portfolio: pl.DataFrame, benchmark: pl.DataFrame, *, column_period: str = "period",
            column_segment: str = "segment", column_weight: str = "weight", column_return: str = "return",
            column_portfolio_period: str | None = None, column_benchmark_period: str | None = None,
            column_portfolio_segment: str | None = None, column_benchmark_segment: str | None = None,
            column_portfolio_weight: str | None = None, column_benchmark_weight: str | None = None,
            column_portfolio_return: str | None = None, column_benchmark_return: str | None = None) -> pl.DataFrame:
    """Align period/segment/weight/return frames, retaining unmatched segments.

    The absent side receives zero weight and the present side's return, which
    assigns no selection to an unheld or off-benchmark segment.
    """
    p_period = column_period if column_portfolio_period is None else column_portfolio_period
    b_period = column_period if column_benchmark_period is None else column_benchmark_period
    p_segment = column_segment if column_portfolio_segment is None else column_portfolio_segment
    b_segment = column_segment if column_benchmark_segment is None else column_benchmark_segment
    p_weight = column_weight if column_portfolio_weight is None else column_portfolio_weight
    b_weight = column_weight if column_benchmark_weight is None else column_benchmark_weight
    p_return = column_return if column_portfolio_return is None else column_portfolio_return
    b_return = column_return if column_benchmark_return is None else column_benchmark_return
    for side, names in ((portfolio, [p_period, p_segment, p_weight, p_return]),
                        (benchmark, [b_period, b_segment, b_weight, b_return])):
        _require(side, names)
        if side.select(pl.struct(names[:2]).is_duplicated().any()).item():
            raise AttributionError("duplicate period/segment")
    # Two side-specific contributions form new, distinct output fields.
    outputs = {"portfolio_weight", "benchmark_weight", "portfolio_return", "benchmark_return"}
    if p_period in outputs or p_segment in outputs:
        raise AttributionError("join keys conflict with output columns")
    p = portfolio.select(p_period, p_segment, pl.col(p_weight).alias("portfolio_weight"),
                         pl.col(p_return).alias("portfolio_return"))
    b = benchmark.select(b_period, b_segment, pl.col(b_weight).alias("benchmark_weight"),
                         pl.col(b_return).alias("benchmark_return"))
    schema = ColumnNames(period=p_period, segment=p_segment)
    d = p.join(b, left_on=[p_period, p_segment], right_on=[b_period, b_segment], how="full",
               coalesce=True).with_columns(
        pl.col(list(schema.weights)).fill_null(0),
        pl.coalesce("portfolio_return", "benchmark_return").alias("portfolio_return"),
        pl.coalesce("benchmark_return", "portfolio_return").alias("benchmark_return"),
    )
    return _validate(d, list(schema.weights + schema.returns), schema).sort([p_period, p_segment])


def roll_up(detail: pl.DataFrame, hierarchy: pl.DataFrame, effects=None, *, column_period: str = "period",
            column_segment: str = "segment", column_child: str = "child",
            column_parent: str = "parent") -> pl.DataFrame:
    """Sum leaves into ancestors. hierarchy has unique child and nullable parent."""
    effects = list(ColumnNames().effects if effects is None else effects)
    _require(detail, [column_period, column_segment] + effects)
    _require(hierarchy, [column_child])
    if column_parent not in hierarchy.columns or column_child == column_parent or hierarchy[
        column_child].n_unique() != hierarchy.height:
        raise AttributionError("hierarchy needs unique children and a distinct parent column")
    parent = dict(hierarchy.select(column_child, column_parent).iter_rows())
    for child in parent:
        seen = set()
        node = child
        while node in parent and parent[node] is not None:
            if node in seen:
                raise AttributionError("hierarchy contains a cycle")
            seen.add(node)
            node = parent[node]
    ancestors = {value for value in parent.values() if value is not None}
    if ancestors.intersection(detail[column_segment].to_list()):
        raise AttributionError("detail must contain leaves, not ancestor rows")
    keys = [column_period, column_segment]
    leaves = detail.select(keys + effects)
    pieces = [leaves]
    current = leaves
    mapping = hierarchy.select(column_child, column_parent)
    while True:
        parent_key = column_parent + "_right" if column_parent in current.columns else column_parent
        joined = current.join(mapping, left_on=column_segment, right_on=column_child, how="inner")
        current = joined.filter(pl.col(parent_key).is_not_null()).select(
            column_period, pl.col(parent_key).alias(column_segment), *effects)
        if current.is_empty():
            break
        pieces.append(current)
    return pl.concat(pieces).group_by(keys).agg(pl.col(effects).sum()).sort(keys)


def _result_columns(result, column_period="period", column_segment="segment",
                    column_portfolio_return="portfolio_return", column_benchmark_return="benchmark_return",
                    column_excess_return="excess_return", column_allocation="allocation",
                    column_selection="selection", column_interaction="interaction"):
    if not isinstance(result, AttributionResult):
        raise TypeError("expected an AttributionResult")
    provided = (column_period, column_segment, column_portfolio_return, column_benchmark_return,
                column_excess_return, column_allocation, column_selection, column_interaction)
    defaults = ("period", "segment", "portfolio_return", "benchmark_return", "excess_return", "allocation", "selection",
                "interaction")
    if provided == defaults:
        return result
    c = result.columns
    # Explicit options describe the existing frames, never rename them.
    c = replace(c, period=column_period, segment=column_segment,
                allocation=column_allocation, selection=column_selection, interaction=column_interaction,
                summary_portfolio_return=column_portfolio_return, summary_benchmark_return=column_benchmark_return,
                summary_excess_return=column_excess_return, summary_allocation=column_allocation,
                summary_selection=column_selection, summary_interaction=column_interaction)
    return AttributionResult(result.detail, result.summary, c)


def _link_inputs(result: AttributionResult) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Validate and prepare chronological periods and compounded totals."""
    c = result.columns
    keys = [c.period, c.segment]
    details = [c.allocation, c.selection, c.interaction]
    effects = [c.summary_allocation, c.summary_selection, c.summary_interaction]
    returns = [c.summary_portfolio_return, c.summary_benchmark_return]
    s = result.summary.sort(c.period)
    _require(s, [c.period] + returns + effects + [c.summary_excess_return])
    if s.height < 2 or s[c.period].n_unique() != s.height:
        raise AttributionError("linking requires at least two unique periods")
    _require(result.detail, keys + details)
    for frame, values in ((s, returns + effects + [c.summary_excess_return]), (result.detail, details)):
        if frame.select(pl.any_horizontal(~pl.col(name).is_finite() for name in values).any()).item():
            raise AttributionError("linked inputs must be finite")
    if result.detail.select(pl.struct(keys).is_duplicated().any()).item():
        raise AttributionError("duplicate period/segment")
    # Generated check columns are private aggregates, not renamed input columns.
    sums = result.detail.group_by(c.period).agg(
        *(pl.col(x).sum().alias("_check_" + str(i)) for i, x in enumerate(details)))
    comparison = s.select(c.period, *effects).join(sums, on=c.period, how="full", coalesce=True)
    if comparison.height != s.height:
        raise AttributionError("detail and summary periods differ")
    for i, name in enumerate(effects):
        check = "_check_" + str(i)
        if comparison.select((pl.col(name).is_null() | pl.col(check).is_null()).any()).item():
            raise AttributionError("detail and summary periods differ")
        _check(comparison, pl.col(name) - pl.col(check))
    if s.select(pl.any_horizontal(pl.col(name) <= -1 for name in returns).any()).item():
        raise AttributionError("linking requires finite period returns greater than -1")
    _check(s, pl.sum_horizontal(effects) - pl.col(c.summary_excess_return))
    _check(s, pl.col(returns[0]) - pl.col(returns[1]) - pl.col(c.summary_excess_return))
    p, b = pl.col(returns[0]), pl.col(returns[1])
    g = p - b
    s = s.with_columns(g.alias("g"), (1 + p).alias("p_growth"), (1 + b).alias("b_growth"))
    totals = s.select((pl.col("p_growth").product() - 1).alias("total_p"),
                      (pl.col("b_growth").product() - 1).alias("total_b"),
                      pl.col("g").sum().alias("sum_g"), (pl.col("g") ** 2).sum().alias("sum_g2"),
                      pl.len().alias("n"))
    return s.join(totals, how="cross").with_columns((pl.col("total_p") - pl.col("total_b")).alias("delta")), totals


def link_carino(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
                column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
                column_excess_return: str = "excess_return", column_allocation: str = "allocation",
                column_selection: str = "selection", column_interaction: str = "interaction") -> AttributionResult:
    """Link arithmetic effects with Cari帽o logarithmic smoothing."""
    result = _result_columns(result, column_period, column_segment, column_portfolio_return,
                             column_benchmark_return, column_excess_return, column_allocation, column_selection,
                             column_interaction)
    p, b, g = pl.col(result.columns.summary_portfolio_return), pl.col(result.columns.summary_benchmark_return), pl.col(
        "g")
    kt = pl.when(g.abs() < 1e-12).then(1 / (1 + b)).otherwise((p.log1p() - b.log1p()) / g)
    delta = pl.col("delta")
    k = pl.when(delta.abs() < 1e-12).then(1 / (1 + pl.col("total_b"))).otherwise(
        (pl.col("total_p").log1p() - pl.col("total_b").log1p()) / delta)
    return _link_arithmetic(result, kt / k)


def link_grap(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
              column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
              column_excess_return: str = "excess_return", column_allocation: str = "allocation",
              column_selection: str = "selection", column_interaction: str = "interaction") -> AttributionResult:
    """Link arithmetic effects using GRAP factors."""
    result = _result_columns(result, column_period, column_segment, column_portfolio_return,
                             column_benchmark_return, column_excess_return, column_allocation, column_selection,
                             column_interaction)
    factor = (pl.col("p_growth").cum_prod().shift(1).fill_null(1) *
              pl.col("b_growth").reverse().cum_prod().shift(1).fill_null(1).reverse())
    return _link_arithmetic(result, factor)


def link_frongello(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
                   column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
                   column_excess_return: str = "excess_return", column_allocation: str = "allocation",
                   column_selection: str = "selection", column_interaction: str = "interaction") -> AttributionResult:
    """Link Frongello effects, using the equivalent GRAP factor form."""
    result = _result_columns(result, column_period, column_segment, column_portfolio_return,
                             column_benchmark_return, column_excess_return, column_allocation, column_selection,
                             column_interaction)
    return link_grap(result)


def link_menchero(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
                  column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
                  column_excess_return: str = "excess_return", column_allocation: str = "allocation",
                  column_selection: str = "selection", column_interaction: str = "interaction") -> AttributionResult:
    """Link arithmetic effects with Menchero optimized smoothing."""
    result = _result_columns(result, column_period, column_segment, column_portfolio_return,
                             column_benchmark_return, column_excess_return, column_allocation, column_selection,
                             column_interaction)
    delta, n = pl.col("delta"), pl.col("n")
    base = 1 + pl.col("total_b")
    m = pl.when(delta.abs() < 1e-12).then(base.pow((n - 1) / n)).otherwise(
        (delta / n) / ((1 + pl.col("total_p")).pow(1 / n) - base.pow(1 / n)))
    correction = pl.when(pl.col("sum_g2") == 0).then(0.0).otherwise(
        (delta - m * pl.col("sum_g")) * pl.col("g") / pl.col("sum_g2"))
    return _link_arithmetic(result, m + correction)


def link_geometric(result: AttributionResult, *, column_period: str = "period", column_segment: str = "segment",
                   column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
                   column_excess_return: str = "excess_return", column_allocation: str = "allocation",
                   column_selection: str = "selection", column_interaction: str = "interaction") -> AttributionResult:
    """Link allocation and selection as geometric relative returns."""
    result = _result_columns(result, column_period, column_segment, column_portfolio_return,
                             column_benchmark_return, column_excess_return, column_allocation, column_selection,
                             column_interaction)
    s, _ = _link_inputs(result)
    b = pl.col(result.columns.summary_benchmark_return)
    if s.select(((1 + b + pl.col(result.columns.summary_allocation)) <= 0).any()).item():
        raise AttributionError("semi-notional wealth must be positive")
    d = s.with_columns((pl.col(result.columns.summary_allocation) / (1 + b)).alias("geometric_allocation"),
                       ((pl.col(result.columns.summary_selection) + pl.col(result.columns.summary_interaction)) / (
                               1 + b + pl.col(result.columns.summary_allocation))).alias("geometric_selection"))
    summary = d.select(
        ((1 + pl.col("geometric_allocation")).product() - 1).alias("allocation"),
        ((1 + pl.col("geometric_selection")).product() - 1).alias("selection"),
        pl.lit(0.0).alias("interaction"), pl.col("total_p").first().alias("portfolio_return"),
        pl.col("total_b").first().alias("benchmark_return"),
        ((1 + pl.col("total_p").first()) / (1 + pl.col("total_b").first()) - 1).alias("excess_return"))
    _check(summary, (1 + pl.col("allocation")) * (1 + pl.col("selection")) - 1 - pl.col("excess_return"))
    return AttributionResult(d, summary, replace(result.columns, summary_portfolio_return="portfolio_return",
                                                 summary_benchmark_return="benchmark_return",
                                                 summary_allocation="allocation", summary_selection="selection",
                                                 summary_interaction="interaction",
                                                 summary_excess_return="excess_return"))


def _link_arithmetic(result: AttributionResult, factor: pl.Expr) -> AttributionResult:
    c = result.columns
    s, totals = _link_inputs(result)
    s = s.with_columns(factor.alias("factor"))
    _check(s.select((pl.col("factor") * pl.col("g")).sum().alias("linked"), pl.col("delta").first()),
           pl.col("linked") - pl.col("delta"))
    detail = result.detail.join(s.select(c.period, "factor"), on=c.period, how="left").with_columns(
        (pl.col(name) * pl.col("factor")).alias("linked_" + out)
        for name, out in zip(c.effects, ColumnNames().effects))
    summary = detail.select(*(pl.col("linked_" + name).sum().alias(name) for name in ColumnNames().effects)).join(
        totals.select(pl.col("total_p").alias("portfolio_return"), pl.col("total_b").alias("benchmark_return"),
                      (pl.col("total_p") - pl.col("total_b")).alias("excess_return")), how="cross")
    _check(summary, pl.sum_horizontal(*ColumnNames().effects) - pl.col("excess_return"))
    schema = replace(c, summary_portfolio_return="portfolio_return", summary_benchmark_return="benchmark_return",
                     summary_allocation="allocation", summary_selection="selection", summary_interaction="interaction",
                     summary_excess_return="excess_return")
    return AttributionResult(detail.sort([c.period, c.segment]), summary, schema)


def currency_premiums(markets: pl.DataFrame, currencies: pl.DataFrame, *,
                      column_period: str = "period", column_segment: str = "segment",
                      column_portfolio_weight: str = "portfolio_weight",
                      column_benchmark_weight: str = "benchmark_weight",
                      column_portfolio_return: str = "portfolio_return",
                      column_benchmark_return: str = "benchmark_return",
                      column_local_cash_return: str = "local_cash_return",
                      market_columns: dict[str, str] | None = None, currency_columns: dict[str, str] | None = None) -> \
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Four-effect log-return market/currency attribution with independent exposures.

    Both inputs use period, segment, portfolio_weight, benchmark_weight.
    Markets additionally need portfolio_return, benchmark_return, local_cash_return;
    currencies need portfolio_return and benchmark_return. Returns are simple.
    """
    common = {"period": column_period, "segment": column_segment, "portfolio_weight": column_portfolio_weight,
              "benchmark_weight": column_benchmark_weight, "portfolio_return": column_portfolio_return,
              "benchmark_return": column_benchmark_return}
    market_mapping = {**common, "local_cash_return": column_local_cash_return}
    for supplied, allowed in ((market_columns, market_mapping), (currency_columns, common)):
        if supplied and set(supplied) - set(allowed):
            raise AttributionError("unknown currency input column role")
    market_mapping = {**market_mapping, **(market_columns or {})}
    currency_mapping = {**common, **(currency_columns or {})}
    mcols = ColumnNames(**{key: market_mapping[key] for key in common})
    ccols = ColumnNames(**currency_mapping)
    cash = market_mapping["local_cash_return"]
    m = _validate(markets,
                  [mcols.portfolio_weight, mcols.benchmark_weight, mcols.portfolio_return, mcols.benchmark_return,
                   cash], mcols)
    c = _validate(currencies,
                  [ccols.portfolio_weight, ccols.benchmark_weight, ccols.portfolio_return, ccols.benchmark_return],
                  ccols)
    if set(m[mcols.period].to_list()) != set(c[ccols.period].to_list()):
        raise AttributionError("market and currency periods differ")
    for data, cols in ((m, [mcols.portfolio_return, mcols.benchmark_return, cash]),
                       (c, [ccols.portfolio_return, ccols.benchmark_return])):
        if data.select(pl.any_horizontal(pl.col(x) <= -1 for x in cols).any()).item():
            raise AttributionError("simple returns must be greater than -1")
    generated = {"p_premium", "b_premium", "p_cash", "b_cash", "market_allocation", "security_selection",
                 "currency_allocation", "hedge_selection", "portfolio_log_contribution", "benchmark_log_contribution"}
    if generated & (set(m.columns) | set(c.columns)):
        raise AttributionError("input columns conflict with generated currency effects")
    m = m.with_columns(
        (pl.col(mcols.portfolio_return).log1p() - pl.col(cash).log1p()).alias("p_premium"),
        (pl.col(mcols.benchmark_return).log1p() - pl.col(cash).log1p()).alias("b_premium"))
    c = c.with_columns(pl.col(ccols.portfolio_return).log1p().alias("p_cash"),
                       pl.col(ccols.benchmark_return).log1p().alias("b_cash"))

    def grid(d, schema, p, b, allocation, selection):
        wp, wb = pl.col(schema.portfolio_weight), pl.col(schema.benchmark_weight)
        return d.with_columns(
            ((wp - wb) * (pl.col(b) - (wb * pl.col(b)).sum().over(schema.period))).alias(allocation),
            (wp * (pl.col(p) - pl.col(b))).alias(selection),
            (wp * pl.col(p)).alias("portfolio_log_contribution"),
            (wb * pl.col(b)).alias("benchmark_log_contribution"))

    m = grid(m, mcols, "p_premium", "b_premium", "market_allocation", "security_selection").sort(
        [mcols.period, mcols.segment])
    c = grid(c, ccols, "p_cash", "b_cash", "currency_allocation", "hedge_selection").sort([ccols.period, ccols.segment])
    ms = m.group_by(mcols.period).agg(pl.col("market_allocation", "security_selection").sum(),
                                      pl.col("portfolio_log_contribution").sum().alias("mp"),
                                      pl.col("benchmark_log_contribution").sum().alias("mb"))
    cs = c.group_by(ccols.period).agg(pl.col("currency_allocation", "hedge_selection").sum(),
                                      pl.col("portfolio_log_contribution").sum().alias("cp"),
                                      pl.col("benchmark_log_contribution").sum().alias("cb"))
    summary = ms.join(cs, left_on=mcols.period, right_on=ccols.period).with_columns(
        (pl.col("mp") + pl.col("cp")).alias("portfolio_log_return"),
        (pl.col("mb") + pl.col("cb")).alias("benchmark_log_return"))
    summary = summary.with_columns(
        (pl.col("portfolio_log_return") - pl.col("benchmark_log_return")).alias("active_log_return")).sort(mcols.period)
    _check(summary, pl.sum_horizontal("market_allocation", "security_selection", "currency_allocation",
                                      "hedge_selection") - pl.col("active_log_return"))
    return m, c, summary


def roll_up_currency_premiums(market_detail: pl.DataFrame, currency_detail: pl.DataFrame, summary: pl.DataFrame, *,
                              column_period: str = "period", column_segment: str = "segment",
                              column_market_allocation: str = "market_allocation",
                              column_security_selection: str = "security_selection",
                              column_currency_allocation: str = "currency_allocation",
                              column_hedge_selection: str = "hedge_selection",
                              column_portfolio_log_return: str = "portfolio_log_return",
                              column_benchmark_log_return: str = "benchmark_log_return",
                              column_active_log_return: str = "active_log_return",
                              column_market_period: str | None = None,
                              column_currency_period: str | None = None, column_market_segment: str | None = None,
                              column_currency_segment: str | None = None):
    """Add log effects through time and return identifier totals and cumulative prefixes."""
    market_effects = [column_market_allocation, column_security_selection]
    currency_effects = [column_currency_allocation, column_hedge_selection]
    all_effects = market_effects + currency_effects
    m_period = column_period if column_market_period is None else column_market_period
    c_period = column_period if column_currency_period is None else column_currency_period
    m_segment = column_segment if column_market_segment is None else column_market_segment
    c_segment = column_segment if column_currency_segment is None else column_currency_segment
    _require(summary, [column_period] + all_effects + [column_portfolio_log_return, column_benchmark_log_return,
                                                       column_active_log_return])
    _require(market_detail, [m_period, m_segment] + market_effects)
    _require(currency_detail, [c_period, c_segment] + currency_effects)
    if summary[column_period].n_unique() != summary.height:
        raise AttributionError("summary periods must be unique")
    for detail, key, segment, effects in ((market_detail, m_period, m_segment, market_effects),
                                          (currency_detail, c_period, c_segment, currency_effects)):
        if detail.select(pl.struct([key, segment]).is_duplicated().any()).item():
            raise AttributionError("duplicate period/segment")
        sums = detail.group_by(key).agg(pl.col(effects).sum())
        comparison = summary.select(column_period, *effects).join(sums, left_on=column_period, right_on=key, how="full",
                                                                  coalesce=True)
        if comparison.select(
                pl.any_horizontal(pl.col(x).is_null() | pl.col(x + "_right").is_null() for x in effects).any()).item():
            raise AttributionError("detail and summary periods differ")
        for effect in effects:
            _check(comparison, pl.col(effect) - pl.col(effect + "_right"))
    _check(summary, pl.sum_horizontal(all_effects) - pl.col(column_active_log_return))
    _check(summary,
           pl.col(column_portfolio_log_return) - pl.col(column_benchmark_log_return) - pl.col(column_active_log_return))
    fields = all_effects + [column_portfolio_log_return, column_benchmark_log_return, column_active_log_return]
    cumulative = summary.sort(column_period).select(column_period, *(pl.col(x).cum_sum().alias(x) for x in fields))
    markets = market_detail.group_by(m_segment).agg(pl.col(market_effects).sum()).sort(m_segment)
    currencies = currency_detail.group_by(c_segment).agg(pl.col(currency_effects).sum()).sort(c_segment)
    return markets, currencies, cumulative
