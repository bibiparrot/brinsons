import pytest
import polars as pl
from brinsons import (AttributionError, fachler, factor_attribution_restricted,
    brinson_interaction_restricted, brinson_interaction_reconciled,
    link_precision_weighted, link_entry_reconciled)


def diagonal(keys, values=None):
    values = values or [1.0] * len(keys)
    return pl.DataFrame([{'row':a,'column':b,'covariance':values[i] if i==j else 0.0}
        for i,a in enumerate(keys) for j,b in enumerate(keys)])


def example():
    return pl.DataFrame({'period':[1,1], 'segment':['one','two'],
      'portfolio_weight':[.6,.4], 'benchmark_weight':[.5,.5],
      'portfolio_return':[.024,-.012], 'benchmark_return':[.02,-.01]})


def test_paper_single_restriction_and_preserved_returns():
    cov=diagonal(['B:one','B:two','P:one','P:two'])
    paper=brinson_interaction_restricted(example(),cov)
    assert paper.summary['allocation'][0] == pytest.approx(.0033)
    assert paper.summary['interaction'][0] == pytest.approx(0,abs=1e-12)
    assert paper.summary['reconciliation_residual'][0] == pytest.approx(.0003)
    exact=brinson_interaction_reconciled(example(),cov)
    assert exact.summary['reconciliation_residual'][0] == pytest.approx(0,abs=1e-12)
    assert exact.summary['interaction'][0] == pytest.approx(0,abs=1e-12)
    d=exact.detail
    assert (d['adjusted_portfolio_return']*d['portfolio_weight']).sum() == pytest.approx(.0096)
    assert (d['adjusted_benchmark_return']*d['benchmark_weight']).sum() == pytest.approx(.005)


def test_factor_restricted():
    data=pl.DataFrame({'asset':['a','b','c'], 'return':[.03,.01,.08],
      'portfolio_weight':[.7,.2,.1], 'benchmark_weight':[.2,.3,.5],
      'intercept':[1.,1.,1.], 'value':[-1.,0.,1.]})
    out=factor_attribution_restricted(data,['intercept','value'])
    assert out.filter(pl.col('factor')=='intercept')['estimate'][0] == pytest.approx(.04)
    assert out.filter(pl.col('factor')=='value')['estimate'][0] == pytest.approx(.025)
    assert out.filter(pl.col('factor')=='value')['adjusted'][0] == pytest.approx(.02)
    active=((data['portfolio_weight']-data['benchmark_weight'])*data['return']).sum()
    assert out['restricted_contribution'].sum() == pytest.approx(active,abs=1e-12)
    assert out['unrestricted_contribution'].sum() != pytest.approx(active,abs=1e-6)
    with pytest.raises(AttributionError): factor_attribution_restricted(data,['intercept','intercept'])


def test_period_weighted_and_entry_reconciliation():
    data=pl.concat([example(),example().with_columns(pl.lit(2,dtype=pl.Int64).alias('period'),
       (pl.col('portfolio_return') * -1).alias('portfolio_return'))])
    periods=fachler(data)
    out=link_precision_weighted(periods,diagonal([1,2],[2.,1.]))
    summary=out.summary.row(0,named=True)
    assert sum(summary[x] for x in ('allocation','selection','interaction')) == pytest.approx(summary['excess_return'],abs=1e-12)
    s=periods.summary.sort('period')
    e=(s['portfolio_return']-s['benchmark_return']).to_list()
    target=summary['excess_return']
    denom=2*e[0]**2+e[1]**2
    expected=1+2*e[0]*(target-sum(e))/denom
    assert out.detail.filter(pl.col('period')==1)['factor'][0] == pytest.approx(expected)
    entries=link_entry_reconciled(periods,diagonal(list(range(12))))
    assert entries.detail['adjusted'].sum() == pytest.approx(target,abs=1e-12)
    assert entries.detail['adjustment'].to_list() == pytest.approx([entries.detail['adjustment'][0]]*12, abs=1e-15)


def test_covariance_guards():
    cov=diagonal(['B:one','B:two','P:one','P:two'])
    with pytest.raises(AttributionError): brinson_interaction_restricted(example(),cov.head(2))
    indefinite=cov.with_columns(pl.when((pl.col('row')=='B:one') & (pl.col('column')=='B:one')).then(-1.).otherwise(pl.col('covariance')).alias('covariance'))
    with pytest.raises(AttributionError): brinson_interaction_restricted(example(),indefinite)


def test_full_covariance_and_baseline():
    d=pl.concat([example(), example().with_columns(pl.lit(2,dtype=pl.Int64).alias('period'),
        (pl.col('portfolio_return') * -1).alias('portfolio_return'))])
    result=fachler(d)
    covariance=pl.DataFrame({'row':[1,1,2,2], 'column':[1,2,1,2], 'covariance':[2.,.4,.4,1.]})
    baseline=pl.DataFrame({'period':[2,1], 'baseline_weight':[.8,1.1]})
    linked=link_precision_weighted(result,covariance,baseline)
    e=result.summary.sort('period')['excess_return'].to_list()
    t=linked.summary['excess_return'][0]
    v=[2*e[0]+.4*e[1],.4*e[0]+e[1]]
    denom=sum(a*b for a,b in zip(e,v))
    expected=[base+v[i]*(t-(1.1*e[0]+.8*e[1]))/denom for i,base in enumerate([1.1,.8])]
    for i in range(2):
        assert linked.detail.filter(pl.col('period')==i+1)['factor'][0] == pytest.approx(expected[i])


def test_degenerate_constraint_and_singular_regression():
    data=example().with_columns(pl.col('benchmark_weight').alias('portfolio_weight'))
    cov=diagonal(['B:one','B:two','P:one','P:two'])
    out=brinson_interaction_restricted(data,cov)
    assert out.summary['interaction'][0] == pytest.approx(0)
    assets=pl.DataFrame({'asset':['a','b'], 'return':[.1,.2], 'portfolio_weight':[.5,.5],
        'benchmark_weight':[.5,.5], 'x':[1.,1.], 'y':[2.,2.]})
    with pytest.raises(AttributionError,match='singular'):
        factor_attribution_restricted(assets,['x','y'])


def test_group_covariance_propagation():
    from brinsons import brinson_group_covariance
    assets=pl.DataFrame({'asset':['a','b','c','d'], 'segment':['one','one','two','two'],
      'portfolio_weight':[.4,.2,.1,.3], 'benchmark_weight':[.25,.25,.25,.25]})
    covariance=diagonal(['a','b','c','d'])
    out=brinson_group_covariance(assets,covariance)
    def get(a,b):
        return out.filter((pl.col('row')==a)&(pl.col('column')==b))['covariance'][0]
    assert get('B:one','B:one') == pytest.approx(.5)
    assert get('P:one','P:one') == pytest.approx(5/9)
    assert get('B:one','P:one') == pytest.approx(.5)
    assert get('B:one','P:two') == pytest.approx(0.)


def test_benchmark_residual_covariance():
    from brinsons import benchmark_residual_covariance, brinson_group_covariance
    assets=pl.DataFrame({'asset':['a','b'], 'segment':['one','one'],
                        'benchmark_weight':[.5,.5], 'portfolio_weight':[.6,.4]})
    residual=benchmark_residual_covariance(assets,diagonal(['a','b']))
    assert residual['covariance'].to_list() == pytest.approx([.5,-.5,-.5,.5])
    mean_cov=brinson_group_covariance(assets,residual)
    assert mean_cov.filter(pl.col('row')=='B:one')['covariance'].to_list() == pytest.approx([0.,0.],abs=1e-15)
