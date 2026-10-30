"""Precision-weighted attribution from Hentschel's Brinson and Factors paper.

Covariance matrices use long Polars frames: row, column, covariance. The
functions accept explicit uncertainty inputs; they do not infer a risk model.
"""
import math
import polars as pl
from .columns import ColumnNames
from .core import AttributionError, AttributionResult, _check, _result_columns, _link_inputs, _require, fachler


def _solve(matrix, rhs):
    """Solve a small nonsingular system with scaled partial pivoting."""
    n = len(rhs)
    a = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, rhs)]
    scale = max((abs(x) for row in matrix for x in row), default=0.0)
    if not math.isfinite(scale) or scale == 0:
        raise AttributionError("singular linear system")
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) <= 1e-12 * scale:
            raise AttributionError("singular or ill-conditioned linear system")
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col+1, n):
            multiple = a[row][col] / a[col][col]
            for j in range(col, n+1):
                a[row][j] -= multiple * a[col][j]
    out = [0.0] * n
    for i in range(n-1, -1, -1):
        out[i] = (a[i][n] - sum(a[i][j] * out[j] for j in range(i+1,n))) / a[i][i]
    return out


def _covariance(covariance, keys, row='row', column='column', value='covariance'):
    _require(covariance, [row,column,value])
    if covariance.select(pl.struct(row,column).is_duplicated().any()).item():
        raise AttributionError('duplicate covariance entry')
    try:
        c = covariance.select(row,column,pl.col(value).cast(pl.Float64))
        if c.select((~pl.col(value).is_finite()).any()).item():
            raise AttributionError('covariance values must be finite')
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise AttributionError('invalid covariance values') from exc
    if set(c[row].to_list()) != set(keys) or set(c[column].to_list()) != set(keys) or c.height != len(keys)**2:
        raise AttributionError('covariance must be a complete square matrix over the exact keys')
    values = {(row, col): value for row,col,value in c.iter_rows()}
    scale = max((abs(x) for x in values.values()), default=0.0)
    tolerance = max(scale * 1e-10, 1e-30)
    if any(abs(values[i,j]-values[j,i]) > tolerance for i in keys for j in keys):
        raise AttributionError('covariance matrix must be symmetric')
    # Semidefinite LDL decomposition, including zero-variance directions.
    lower = [[0.0]*len(keys) for _ in keys]
    diagonal = [0.0]*len(keys)
    for i in range(len(keys)):
        residual = values[keys[i],keys[i]] - sum(lower[i][k]**2 * diagonal[k] for k in range(i))
        if residual < -tolerance:
            raise AttributionError('covariance matrix must be positive semidefinite')
        diagonal[i] = max(0.0, residual)
        for j in range(i+1,len(keys)):
            v = values[keys[j],keys[i]] - sum(lower[j][k]*lower[i][k]*diagonal[k] for k in range(i))
            if diagonal[i] <= tolerance:
                if abs(v) > tolerance:
                    raise AttributionError('covariance matrix must be positive semidefinite')
            else:
                lower[j][i] = v/diagonal[i]
    return c


def _restricted(vector, covariance, constraints, targets, *, column_covariance_row='row',
                column_covariance_column='column', column_covariance_value='covariance'):
    """Apply a restricted GLS update with named vector entries."""
    keys = vector['key'].to_list()
    c = _covariance(covariance, keys, column_covariance_row, column_covariance_column, column_covariance_value)
    q = vector.select('key', *[pl.Series(f'q{i}', values) for i,values in enumerate(constraints)])
    contributions = c.join(q, left_on=column_covariance_column, right_on='key', how='left')
    cq = contributions.group_by(column_covariance_row).agg(*[(pl.col(column_covariance_value)*pl.col(f'q{i}')).sum().alias(f'cq{i}') for i in range(len(targets))])
    joined = vector.join(q, on='key').join(cq, left_on='key', right_on=column_covariance_row)
    gram = [[joined.select((pl.col(f'q{i}')*pl.col(f'cq{j}')).sum()).item() for j in range(len(targets))] for i in range(len(targets))]
    residual = [target-joined.select((pl.col(f'q{i}')*pl.col('estimate')).sum()).item() for i,target in enumerate(targets)]
    if all(abs(x) <= 1e-14 for x in residual):
        multiplier = [0.0]*len(targets)
    else:
        multiplier = _solve(gram, residual)
    return joined.with_columns((pl.col('estimate')+pl.sum_horizontal(*[pl.col(f'cq{i}')*m for i,m in enumerate(multiplier)])).alias('adjusted'))


def factor_attribution_restricted(assets: pl.DataFrame, factors: list[str], *, column_asset: str = "asset",
        column_return: str = "return", column_portfolio_weight: str = "portfolio_weight",
        column_benchmark_weight: str = "benchmark_weight", column_regression_weight: str = "regression_weight") -> pl.DataFrame:
    """Cross-sectional weighted regression and paper's active-exact factor update.

    assets contains asset, return, portfolio_weight, benchmark_weight, and one
    exposure column per factor. Optional regression_weight defaults to one.
    """
    if not factors or len(set(factors)) != len(factors):
        raise AttributionError('factor columns must be nonempty and unique')
    names = [column_asset, column_return, column_portfolio_weight, column_benchmark_weight]
    if set(factors) & set(names):
        raise AttributionError('factor columns must be distinct from other input roles')
    _require(assets, names + factors)
    if assets[column_asset].n_unique() != assets.height:
        raise AttributionError('asset identifiers must be unique')
    if column_regression_weight != 'regression_weight' or column_regression_weight in assets.columns:
        _require(assets, [column_regression_weight])
        w = pl.col(column_regression_weight)
        cols = names[1:] + factors + [column_regression_weight]
    else:
        w = pl.lit(1.0)
        cols = names[1:] + factors
    _require(assets, cols)
    try:
        assets = assets.with_columns(pl.col(list(dict.fromkeys(cols))).cast(pl.Float64))
        if assets.select(pl.any_horizontal(~pl.col(x).is_finite() for x in cols).any()).item():
            raise AttributionError('factor inputs must be finite')
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise AttributionError('factor inputs must be numeric and finite') from exc
    if assets.select((w <= 0).any()).item():
        raise AttributionError('regression weights must be positive')
    for weight in [column_portfolio_weight,column_benchmark_weight]:
        if abs(assets[weight].sum()-1) > 1e-8:
            raise AttributionError('portfolio and benchmark weights must sum to one')
    cross = [[assets.select((w*pl.col(i)*pl.col(j)).sum()).item() for j in factors] for i in factors]
    rhs = [assets.select((w*pl.col(i)*pl.col(column_return)).sum()).item() for i in factors]
    estimates = _solve(cross,rhs)
    inverse_cols = [_solve(cross, [float(i==j) for i in range(len(factors))]) for j in range(len(factors))]
    covariance = pl.DataFrame([{'row':i,'column':j,'covariance':inverse_cols[k][l]} for l,i in enumerate(factors) for k,j in enumerate(factors)])
    active = pl.col(column_portfolio_weight)-pl.col(column_benchmark_weight)
    exposure = [assets.select((active*pl.col(f)).sum()).item() for f in factors]
    active_return = assets.select((active*pl.col(column_return)).sum()).item()
    vector = pl.DataFrame({'key':factors,'estimate':estimates})
    adjusted = _restricted(vector,covariance,[exposure],[active_return]).with_columns(
        (pl.col('q0')*pl.col('estimate')).alias('unrestricted_contribution'),
        (pl.col('q0')*pl.col('adjusted')).alias('restricted_contribution'))
    _check(adjusted.select(pl.col('restricted_contribution').sum().alias('total')), pl.col('total')-active_return)
    return adjusted.select(pl.col('key').alias('factor'),pl.col('q0').alias('active_exposure'),
                           'estimate','adjusted','unrestricted_contribution','restricted_contribution')


def brinson_interaction_restricted(data: pl.DataFrame, covariance: pl.DataFrame, *,
        column_period: str = "period", column_segment: str = "segment", column_portfolio_weight: str = "portfolio_weight",
        column_benchmark_weight: str = "benchmark_weight", column_portfolio_return: str = "portfolio_return",
        column_benchmark_return: str = "benchmark_return", column_covariance_row: str = "row",
        column_covariance_column: str = "column", column_covariance_value: str = "covariance") -> AttributionResult:
    """Apply the paper's single interaction constraint to one period.

    The paper's restriction alone may change the implied active return; inspect
    summary.reconciliation_residual. For exact observed returns use
    brinson_interaction_reconciled, which adds return-preservation constraints.
    """
    original = fachler(data, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return)
    return _brinson_restricted(original, covariance, False,
        column_covariance_row, column_covariance_column, column_covariance_value)


def brinson_interaction_reconciled(data: pl.DataFrame, covariance: pl.DataFrame, *,
        column_period: str = "period", column_segment: str = "segment", column_portfolio_weight: str = "portfolio_weight",
        column_benchmark_weight: str = "benchmark_weight", column_portfolio_return: str = "portfolio_return",
        column_benchmark_return: str = "benchmark_return", column_covariance_row: str = "row",
        column_covariance_column: str = "column", column_covariance_value: str = "covariance") -> AttributionResult:
    """Remove total interaction while preserving both original side returns."""
    original = fachler(data, column_period=column_period, column_segment=column_segment,
        column_portfolio_weight=column_portfolio_weight, column_benchmark_weight=column_benchmark_weight,
        column_portfolio_return=column_portfolio_return, column_benchmark_return=column_benchmark_return)
    return _brinson_restricted(original, covariance, True,
        column_covariance_row, column_covariance_column, column_covariance_value)


def _brinson_restricted(original, covariance, preserve_returns, covariance_row, covariance_column, covariance_value):
    c = original.columns
    if original.summary.height != 1:
        raise AttributionError('restricted Brinson accepts one period per call')
    d = original.detail.sort(c.segment)
    if d[c.segment].cast(pl.String).n_unique() != d.height:
        raise AttributionError('segment string labels must be unique')
    labels = d[c.segment].cast(pl.String).to_list()
    keys = ['B:'+x for x in labels]+['P:'+x for x in labels]
    wb, wp = d[c.benchmark_weight].to_list(), d[c.portfolio_weight].to_list()
    active = [p-b for p,b in zip(wp,wb)]
    initial = d[c.benchmark_return].to_list()+d[c.portfolio_return].to_list()
    vector = pl.DataFrame({'key': keys, 'estimate': initial})
    constraints = [[-x for x in active]+active]
    targets = [0.0]
    if preserve_returns:
        constraints += [wb+[0.0]*len(wb), [0.0]*len(wp)+wp]
        targets += [original.summary[c.summary_benchmark_return][0],original.summary[c.summary_portfolio_return][0]]
    adjusted = _restricted(vector,covariance,constraints,targets,
        column_covariance_row=covariance_row, column_covariance_column=covariance_column,
        column_covariance_value=covariance_value)
    b = adjusted.filter(pl.col('key').str.starts_with('B:')).select(pl.col('key').str.slice(2).alias('_label'),pl.col('adjusted').alias('adjusted_benchmark_return'))
    p = adjusted.filter(pl.col('key').str.starts_with('P:')).select(pl.col('key').str.slice(2).alias('_label'),pl.col('adjusted').alias('adjusted_portfolio_return'))
    d = d.with_columns(pl.col(c.segment).cast(pl.String).alias('_label')).join(b,on='_label').join(p,on='_label')
    dw = pl.col(c.portfolio_weight)-pl.col(c.benchmark_weight)
    detail = d.with_columns(
        (dw*(pl.col('adjusted_benchmark_return')-pl.col(c.benchmark_return).mul(pl.col(c.benchmark_weight)).sum())).alias(c.allocation),
        (pl.col(c.benchmark_weight)*(pl.col('adjusted_portfolio_return')-pl.col('adjusted_benchmark_return'))).alias(c.selection),
        (dw*(pl.col('adjusted_portfolio_return')-pl.col('adjusted_benchmark_return'))).alias(c.interaction)).drop('_label').sort(list(c.keys))
    sums = detail.select(pl.col(list(c.effects)).sum())
    observed = original.summary.select(c.period,c.summary_portfolio_return,c.summary_benchmark_return,c.summary_excess_return)
    summary = observed.join(sums,how='cross').with_columns(
        (pl.col(c.summary_excess_return)-pl.col(c.allocation)-pl.col(c.selection)).alias('reconciliation_residual'))
    if abs(summary[c.interaction][0]) > 1e-9:
        raise AttributionError('interaction constraint did not reconcile')
    if preserve_returns:
        _check(summary,pl.col('reconciliation_residual'))
    return AttributionResult(detail,summary,c)


def link_precision_weighted(result: AttributionResult, covariance: pl.DataFrame, baseline: pl.DataFrame | None = None, *,
        column_period: str = "period", column_segment: str = "segment", column_portfolio_return: str = "portfolio_return",
        column_benchmark_return: str = "benchmark_return", column_excess_return: str = "excess_return",
        column_allocation: str = "allocation", column_selection: str = "selection", column_interaction: str = "interaction",
        column_covariance_row: str = "row", column_covariance_column: str = "column", column_covariance_value: str = "covariance",
        column_baseline_period: str | None = None, column_baseline_weight: str = "baseline_weight") -> AttributionResult:
    """Apply equation (33) with a period covariance and optional baseline weights.

    Covariance row/column keys equal the period values; baseline, if supplied,
    has unique period and baseline_weight columns.
    """
    result = _result_columns(result, column_period, column_segment, column_portfolio_return, column_benchmark_return,
        column_excess_return, column_allocation, column_selection, column_interaction)
    s, totals = _link_inputs(result)
    period_key = result.columns.period
    if baseline is None:
        estimate = pl.lit(1.0)
    else:
        baseline_period = period_key if column_baseline_period is None else column_baseline_period
        _require(baseline,[baseline_period,column_baseline_weight])
        if baseline[baseline_period].n_unique() != baseline.height or set(baseline[baseline_period].to_list()) != set(s[period_key].to_list()):
            raise AttributionError('baseline must contain each period exactly once')
        s = s.join(baseline.select(baseline_period,column_baseline_weight),left_on=period_key,right_on=baseline_period)
        estimate = pl.col(column_baseline_weight)
    if s.select((~estimate.cast(pl.Float64).is_finite()).any()).item():
        raise AttributionError('baseline weights must be finite')
    vector = s.select(pl.col(period_key).alias('key'),estimate.cast(pl.Float64).alias('estimate'))
    adjusted = _restricted(vector,covariance,[s['g'].to_list()],[s['delta'][0]],
        column_covariance_row=column_covariance_row, column_covariance_column=column_covariance_column,
        column_covariance_value=column_covariance_value)
    c = result.columns
    detail = result.detail.join(adjusted.select('key',pl.col('adjusted').alias('factor')),left_on=period_key,right_on='key',how='left').with_columns(
        (pl.col(name)*pl.col('factor')).alias('linked_'+out) for name,out in zip(c.effects,ColumnNames().effects))
    summary = detail.select(*(pl.col('linked_'+name).sum().alias(name) for name in ColumnNames().effects)).join(
        totals.select(pl.col('total_p').alias('portfolio_return'),pl.col('total_b').alias('benchmark_return'),
            (pl.col('total_p')-pl.col('total_b')).alias('excess_return')),how='cross')
    _check(summary,pl.sum_horizontal(*ColumnNames().effects)-pl.col('excess_return'))
    return AttributionResult(detail.sort(list(c.keys)),summary,c)


def link_entry_reconciled(result: AttributionResult, covariance: pl.DataFrame, *, column_period: str = "period",
        column_segment: str = "segment", column_portfolio_return: str = "portfolio_return", column_benchmark_return: str = "benchmark_return",
        column_excess_return: str = "excess_return", column_allocation: str = "allocation", column_selection: str = "selection",
        column_interaction: str = "interaction", column_covariance_row: str = "row", column_covariance_column: str = "column",
        column_covariance_value: str = "covariance") -> AttributionResult:
    """Apply equation (37) to individual segment/effect entries.

    Covariance keys are integer entry_id values in sorted period/segment order,
    with allocation, selection, interaction in that order; output includes them.
    """
    result = _result_columns(result, column_period, column_segment, column_portfolio_return, column_benchmark_return,
        column_excess_return, column_allocation, column_selection, column_interaction)
    _, totals = _link_inputs(result)
    c = result.columns
    keys = [c.period,c.segment]
    effects = [c.allocation,c.selection,c.interaction]
    d = result.detail.sort(keys).select(keys+effects).unpivot(index=keys,on=effects,variable_name='effect',value_name='estimate')
    # Polars unpivot is column-major; explicitly establish period/segment/effect order.
    d = d.with_columns(pl.col('effect').replace_strict(dict(zip(effects,range(3)))).alias('_order')).sort(keys+['_order']).drop('_order').with_row_index('entry_id')
    vector = d.select(pl.col('entry_id').alias('key'),'estimate')
    target = totals['total_p'][0]-totals['total_b'][0]
    adjusted = _restricted(vector,covariance,[[1.0]*d.height],[target],
        column_covariance_row=column_covariance_row, column_covariance_column=column_covariance_column,
        column_covariance_value=column_covariance_value)
    detail = d.join(adjusted.select(pl.col('key').alias('entry_id'),'adjusted'),on='entry_id').with_columns((pl.col('adjusted')-pl.col('estimate')).alias('adjustment')).sort('entry_id')
    summary = detail.select(*(pl.col('adjusted').filter(pl.col('effect') == name).sum().alias(out) for name,out in zip(effects,ColumnNames().effects))).join(
        totals.select(pl.col('total_p').alias('portfolio_return'),pl.col('total_b').alias('benchmark_return'),
        (pl.col('total_p')-pl.col('total_b')).alias('excess_return')),how='cross')
    _check(summary,pl.sum_horizontal(*ColumnNames().effects)-pl.col('excess_return'))
    return AttributionResult(detail,summary,c)


def brinson_group_covariance(assets: pl.DataFrame, residual_covariance: pl.DataFrame, *, column_asset: str = "asset",
        column_segment: str = "segment", column_portfolio_weight: str = "portfolio_weight", column_benchmark_weight: str = "benchmark_weight",
        column_covariance_row: str = "row", column_covariance_column: str = "column", column_covariance_value: str = "covariance") -> pl.DataFrame:
    """Propagate asset residual covariance to stacked benchmark/portfolio group means.

    assets has unique asset, segment, portfolio_weight, benchmark_weight.
    The covariance is a complete matrix keyed by asset IDs; output keys use
    B:<segment> and P:<segment>. Each group must have nonzero weight on both sides.
    """
    _require(assets,[column_asset,column_segment,column_portfolio_weight,column_benchmark_weight])
    if assets[column_asset].n_unique() != assets.height:
        raise AttributionError('asset identifiers must be unique')
    try:
        a = assets.with_columns(pl.col(column_portfolio_weight,column_benchmark_weight).cast(pl.Float64), pl.col(column_segment).cast(pl.String))
        if a.select(pl.any_horizontal(~pl.col(c).is_finite() for c in [column_portfolio_weight,column_benchmark_weight]).any()).item():
            raise AttributionError('weights must be finite')
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise AttributionError('weights must be numeric and finite') from exc
    covariance = _covariance(residual_covariance,a[column_asset].to_list(),column_covariance_row,column_covariance_column,column_covariance_value)
    a = a.with_columns(pl.col(column_benchmark_weight).sum().over(column_segment).alias('_b_sum'),
                       pl.col(column_portfolio_weight).sum().over(column_segment).alias('_p_sum'))
    if a.select(((pl.col('_b_sum').abs() <= 1e-14) | (pl.col('_p_sum').abs() <= 1e-14)).any()).item():
        raise AttributionError('each group needs nonzero benchmark and portfolio weight')
    b = a.select(pl.concat_str(pl.lit('B:'),pl.col(column_segment)).alias('key'),column_asset,
                 (pl.col(column_benchmark_weight)/pl.col('_b_sum')).alias('coefficient'))
    p = a.select(pl.concat_str(pl.lit('P:'),pl.col(column_segment)).alias('key'),column_asset,
                 (pl.col(column_portfolio_weight)/pl.col('_p_sum')).alias('coefficient'))
    coefficients = pl.concat([b,p])
    rows = coefficients.join(covariance,left_on=column_asset,right_on=column_covariance_row,how='inner')
    other = coefficients.select(pl.col('key').alias('other_key'),pl.col(column_asset).alias('other_asset'),
                                pl.col('coefficient').alias('other_coefficient'))
    rows = rows.join(other,left_on=column_covariance_column,right_on='other_asset',how='inner')
    propagated = rows.group_by('key','other_key').agg(
        (pl.col('coefficient')*pl.col(column_covariance_value)*pl.col('other_coefficient')).sum().alias('covariance'))
    keys = coefficients.select('key').unique().sort('key')
    grid = keys.select(pl.col('key').alias('row')).join(keys.select(pl.col('key').alias('column')),how='cross')
    return grid.join(propagated,left_on=['row','column'],right_on=['key','other_key'],how='left').with_columns(
        pl.col('covariance').fill_null(0.0)).sort('row','column')


def benchmark_residual_covariance(assets: pl.DataFrame, return_covariance: pl.DataFrame, *, column_asset: str = "asset",
        column_segment: str = "segment", column_benchmark_weight: str = "benchmark_weight", column_covariance_row: str = "row",
        column_covariance_column: str = "column", column_covariance_value: str = "covariance") -> pl.DataFrame:
    """Construct paper equation (23): benchmark residual covariance M Sigma M'.

    Input assets has unique asset, segment, and benchmark_weight. Return
    covariance is a complete long matrix indexed by the asset IDs.
    """
    _require(assets,[column_asset,column_segment,column_benchmark_weight])
    if assets[column_asset].n_unique() != assets.height:
        raise AttributionError('asset identifiers must be unique')
    try:
        a = assets.select(column_asset,column_segment,pl.col(column_benchmark_weight).cast(pl.Float64))
        if a.select((~pl.col(column_benchmark_weight).is_finite()).any()).item():
            raise AttributionError('benchmark weights must be finite')
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise AttributionError('benchmark weights must be numeric and finite') from exc
    a = a.with_columns(pl.col(column_benchmark_weight).sum().over(column_segment).alias('group_weight'))
    if a.select((pl.col('group_weight').abs() <= 1e-14).any()).item():
        raise AttributionError('benchmark group weights must be nonzero')
    covariance = _covariance(return_covariance,a[column_asset].to_list(),column_covariance_row,column_covariance_column,column_covariance_value)
    residual_maker = a.select(pl.col(column_asset).alias('out'), pl.col(column_segment).alias('out_segment')).join(
        a.select(column_asset,column_segment,column_benchmark_weight,'group_weight'),how='cross').select(
        'out',column_asset,((pl.col('out')==pl.col(column_asset)).cast(pl.Float64)-
          pl.when(pl.col('out_segment')==pl.col(column_segment)).then(pl.col(column_benchmark_weight)/pl.col('group_weight')).otherwise(0.0)).alias('coefficient'))
    other = residual_maker.select(pl.col('out').alias('other_out'),pl.col(column_asset).alias('other_asset'),
                                  pl.col('coefficient').alias('other_coefficient'))
    products = residual_maker.join(covariance,left_on=column_asset,right_on=column_covariance_row,how='inner').join(
        other,left_on=column_covariance_column,right_on='other_asset',how='inner')
    return products.group_by('out','other_out').agg(
        (pl.col('coefficient')*pl.col(column_covariance_value)*pl.col('other_coefficient')).sum().alias('covariance')
    ).select(pl.col('out').alias('row'),pl.col('other_out').alias('column'),'covariance').sort('row','column')
