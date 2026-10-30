import random
import pytest
import polars as pl
import pybrinson as ref
from brinsons import (AttributionError, bhb, fachler, currency_attribution,
    link_carino, link_grap, link_frongello, link_menchero, link_geometric, prepare, roll_up)


def generated():
    rng = random.Random(461)
    rows = []
    for t in range(8):
        p = [rng.uniform(.1, 1) for _ in range(5)]
        b = [rng.uniform(.1, 1) for _ in range(5)]
        for i in range(5):
            local, fx = rng.uniform(-.12, .18), rng.uniform(-.04, .04)
            rows.append(dict(period=t, segment=f"s{i}", portfolio_weight=p[i]/sum(p), benchmark_weight=b[i]/sum(b),
                             portfolio_return=local+fx, benchmark_return=rng.uniform(-.1,.15)+fx,
                             local_return=local, currency_return=fx))
    return pl.DataFrame(rows)


def reference(data, method):
    result=[]
    for t in data['period'].unique().sort().to_list():
        rows=data.filter(pl.col('period') == t).to_dicts()
        segs=[ref.Segment(r['segment'], r['portfolio_weight'], r['benchmark_weight'],
              r['portfolio_return'],r['benchmark_return'],local_return=r['local_return'],currency_return=r['currency_return']) for r in rows]
        result.append(getattr(ref,method)(segs,period=str(t)))
    return result


@pytest.mark.parametrize('function,fn', [(bhb,'bhb'),(fachler,'fachler')])
def test_single_period_against_reference(function, fn):
    data=generated()
    actual=function(data)
    expected=reference(data, fn)
    for t, want in enumerate(expected):
        summary=actual.summary.filter(pl.col('period') == t).row(0, named=True)
        for col in ['portfolio_return','benchmark_return','excess_return','allocation','selection','interaction']:
            assert summary[col] == pytest.approx(getattr(want,col),abs=1e-12)
        detail=actual.detail.filter(pl.col('period') == t)
        for seg in want.by_segment:
            got=detail.filter(pl.col('segment') == seg.name).row(0,named=True)
            for col in ['allocation','selection','interaction']:
                assert got[col] == pytest.approx(getattr(seg,col), abs=1e-12)
    two=function(data, effects=2).summary
    for a,b in zip(actual.summary.iter_rows(named=True),two.iter_rows(named=True)):
        assert b['selection'] == pytest.approx(a['selection']+a['interaction'])


@pytest.mark.parametrize('function,name', [(link_carino,'carino'),(link_grap,'grap'),(link_frongello,'frongello'),(link_menchero,'menchero'),(link_geometric,'geometric')])
def test_linking_against_reference(function,name):
    data=generated()
    refs=reference(data,'fachler')
    actual=function(fachler(data)).summary.row(0,named=True)
    expected=getattr(ref,'link_'+name)(refs)
    for col in ['portfolio_return','benchmark_return','excess_return','allocation','selection','interaction']:
        assert actual[col] == pytest.approx(getattr(expected,col),abs=1e-11)


def test_currency_against_reference():
    data=generated()
    actual=currency_attribution(data)
    for t, expected in enumerate(reference(data,'currency_attribution')):
        got=actual.summary.filter(pl.col('period') == t).row(0,named=True)
        for col in ['portfolio_return','benchmark_return','excess_return','market_allocation','security_selection','currency_allocation','interaction']:
            assert got[col] == pytest.approx(getattr(expected,col),abs=1e-12)
        for seg in expected.by_segment:
            row=actual.detail.filter((pl.col('period') == t)&(pl.col('segment') == seg.name)).row(0,named=True)
            for col in ['market_allocation','security_selection','currency_allocation','interaction']:
                assert row[col] == pytest.approx(getattr(seg,col),abs=1e-12)


def test_equal_cumulative_menchero():
    data=pl.DataFrame({'period':[1,2], 'segment':['x','x'], 'portfolio_weight':[1.,1.], 'benchmark_weight':[1.,1.],
                       'portfolio_return':[.1,-.1], 'benchmark_return':[0.,-.01]})
    r=link_menchero(fachler(data)).summary.row(0,named=True)
    assert r['excess_return'] == pytest.approx(0,abs=1e-12)
    assert r['allocation']+r['selection']+r['interaction'] == pytest.approx(0,abs=1e-12)


def test_rollup_and_prepare():
    data=generated()
    p=data.select('period','segment',pl.col('portfolio_weight').alias('weight'),pl.col('portfolio_return').alias('return'))
    b=data.select('period','segment',pl.col('benchmark_weight').alias('weight'),pl.col('benchmark_return').alias('return'))
    ready=prepare(p,b)
    result=fachler(ready)
    tree=pl.DataFrame({'child':['s0','s1','group'], 'parent':['group','group','root']})
    rolled=roll_up(result.detail,tree)
    for t in range(8):
        d=rolled.filter(pl.col('period')==t)
        for c in ['allocation','selection','interaction']:
            assert d.filter(pl.col('segment')=='root')[c][0] == pytest.approx(d.filter(pl.col('segment').is_in(['s0','s1']))[c].sum())
    with pytest.raises(AttributionError):
        roll_up(result.detail,pl.DataFrame({'child':['a','b'],'parent':['b','a']}))


def test_reject_invalid():
    data=generated()
    with pytest.raises(AttributionError): fachler(pl.concat([data,data.head(1)]))
    with pytest.raises(AttributionError): fachler(data.with_columns(pl.lit(float('nan')).alias('portfolio_return')))
    with pytest.raises(AttributionError): currency_attribution(data.with_columns(pl.lit(1.).alias('local_return')))
    with pytest.raises(AttributionError): link_carino(fachler(data.filter(pl.col('period')==0)))


def test_independent_currency_premiums():
    from math import expm1
    from brinsons import currency_premiums
    markets=pl.DataFrame({'period':[1,1], 'segment':['Equity','Bonds'],
        'portfolio_weight':[.6,.4], 'benchmark_weight':[.5,.5],
        'portfolio_return':[expm1(.08),expm1(.02)], 'benchmark_return':[expm1(.05),expm1(.03)],
        'local_cash_return':[0.,0.]})
    currencies=pl.DataFrame({'period':[1,1], 'segment':['EUR','USD'],
        'portfolio_weight':[.7,.3], 'benchmark_weight':[.5,.5],
        'portfolio_return':[expm1(.04),expm1(.015)], 'benchmark_return':[expm1(.03),expm1(.01)]})
    m,c,s=currency_premiums(markets,currencies)
    row=s.row(0,named=True)
    assert row['active_log_return'] == pytest.approx(.0285)
    assert sum(row[k] for k in ['market_allocation','security_selection','currency_allocation','hedge_selection']) == pytest.approx(.0285)
    assert m.height==2 and c.height==2


def test_link_rejects_inconsistent_detail():
    from brinsons import AttributionResult
    result=fachler(generated())
    broken=result.detail.with_columns((pl.col('selection')+.01).alias('selection'))
    with pytest.raises(AttributionError): link_carino(AttributionResult(broken,result.summary))


def test_cumulative_and_segment_effects_reconcile():
    result=bhb(generated())
    for function in [link_carino,link_grap,link_frongello,link_menchero]:
        linked=function(result)
        for effect in ['allocation','selection','interaction']:
            assert linked.detail['linked_'+effect].sum() == pytest.approx(linked.summary[effect][0],abs=1e-12)


def test_hierarchy_against_pybrinson():
    data=generated().filter(pl.col('period')==0)
    tree=pl.DataFrame({'child':['s0','s1','s2','s3','s4','a','b'],
                       'parent':['a','a','b','b','b','root','root']})
    result=roll_up(fachler(data).detail,tree)
    segments=[ref.Segment(r['segment'],r['portfolio_weight'],r['benchmark_weight'],
                          r['portfolio_return'],r['benchmark_return'],parent='a' if r['segment'] in ('s0','s1') else 'b')
              for r in data.to_dicts()]
    expected=ref.fachler(segments,parents={'a':'root','b':'root','root':None})
    for item in expected.by_segment:
        actual=result.filter(pl.col('segment')==item.name).row(0,named=True)
        for col in ['allocation','selection','interaction']:
            assert actual[col] == pytest.approx(getattr(item,col),abs=1e-12)


def test_currency_rollup():
    from math import expm1
    from brinsons import currency_premiums, roll_up_currency_premiums
    markets=pl.DataFrame({'period':[1,2], 'segment':['World','World'],
        'portfolio_weight':[1.,1.], 'benchmark_weight':[1.,1.],
        'portfolio_return':[expm1(.08),expm1(-.02)], 'benchmark_return':[expm1(.05),expm1(.01)],
        'local_cash_return':[0.,0.]})
    currencies=pl.DataFrame({'period':[1,2], 'segment':['USD','USD'],
        'portfolio_weight':[1.,1.], 'benchmark_weight':[1.,1.],
        'portfolio_return':[expm1(.04),expm1(.015)], 'benchmark_return':[expm1(.03),expm1(.01)]})
    m,c,s=currency_premiums(markets,currencies)
    mt,ct,cumulative=roll_up_currency_premiums(m,c,s)
    assert cumulative['active_log_return'][-1] == pytest.approx(.015)
    assert mt['security_selection'][0] + ct['hedge_selection'][0] == pytest.approx(.015)
