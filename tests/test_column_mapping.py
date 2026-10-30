import polars as pl
import pytest
from polars.testing import assert_frame_equal
from brinsons import (AttributionError, AttributionResult, bhb, fachler, prepare, roll_up,
    currency_attribution, currency_premiums, roll_up_currency_premiums,
    link_carino, link_grap, link_frongello, link_menchero, link_geometric,
    factor_attribution_restricted, brinson_interaction_restricted, brinson_interaction_reconciled,
    brinson_group_covariance, benchmark_residual_covariance, link_precision_weighted, link_entry_reconciled)
from brinsons import ColumnNames


def facts():
    return pl.DataFrame({'period':[1,1,2,2], 'segment':['a','b','a','b'],
        'portfolio_weight':[.6,.4,.55,.45], 'benchmark_weight':[.5,.5,.5,.5],
        'portfolio_return':[.05,.01,.03,.02], 'benchmark_return':[.04,.02,.025,.015],
        'local_return':[.04,.02,.02,.01], 'currency_return':[.01,-.01,.01,.01]})


MAPPING={'period':'date','segment':'sector','portfolio_weight':'pw','benchmark_weight':'bw',
         'portfolio_return':'pr','benchmark_return':'br','local_return':'local','currency_return':'fx'}
ARGS={'column_'+k:v for k,v in MAPPING.items()}

@pytest.mark.parametrize('fn',[bhb,fachler])
def test_single_period_custom_columns(fn):
    df=facts()
    opts={k:v for k,v in ARGS.items() if k not in ('column_local_return','column_currency_return')}
    actual=fn(df.rename({k:v for k,v in MAPPING.items() if k not in ('local_return','currency_return')}),**opts)
    expected=fn(df)
    inverse={v:k for k,v in MAPPING.items() if k not in ('local_return','currency_return')}
    assert set(inverse).issubset(actual.detail.columns)
    assert_frame_equal(actual.detail.rename(inverse),expected.detail)
    assert_frame_equal(actual.summary.rename({'date':'period'}),expected.summary)
    assert actual.columns.period == 'date' and actual.columns.segment == 'sector'


def test_currency_custom_columns():
    df=facts()
    expected=currency_attribution(df)
    actual=currency_attribution(df.rename(MAPPING),**ARGS)
    assert_frame_equal(actual.summary.rename({'date':'period'}),expected.summary)
    assert_frame_equal(actual.detail.rename({v:k for k,v in MAPPING.items()}),expected.detail)


def test_prepare_different_side_names_and_hierarchy():
    df=facts()
    p=df.select(pl.col('period').alias('p_date'),pl.col('segment').alias('p_sector'),pl.col('portfolio_weight').alias('p_weight'),pl.col('portfolio_return').alias('p_return'))
    b=df.select(pl.col('period').alias('b_date'),pl.col('segment').alias('b_sector'),pl.col('benchmark_weight').alias('b_weight'),pl.col('benchmark_return').alias('b_return'))
    result=prepare(p,b,column_portfolio_period='p_date',column_portfolio_segment='p_sector',column_portfolio_weight='p_weight',column_portfolio_return='p_return',
        column_benchmark_period='b_date',column_benchmark_segment='b_sector',column_benchmark_weight='b_weight',column_benchmark_return='b_return')
    expected_join=df.rename({'period':'p_date','segment':'p_sector'}).select(result.columns)
    assert_frame_equal(result,expected_join)
    result_detail=fachler(result,columns=ColumnNames(period='p_date',segment='p_sector')).detail
    detail=result_detail.rename({'p_date':'date','p_sector':'sector'})
    tree=pl.DataFrame({'node':['a','b','root'],'parent_node':['root','root',None]})
    actual=roll_up(detail,tree,column_period='date',column_segment='sector',column_child='node',column_parent='parent_node')
    expected=roll_up(result_detail,tree.rename({'node':'child','parent_node':'parent'}),column_period='p_date',column_segment='p_sector')
    assert_frame_equal(actual.rename({'date':'p_date','sector':'p_sector'}),expected)


@pytest.mark.parametrize('fn',[link_carino,link_grap,link_frongello,link_menchero,link_geometric])
def test_link_custom_result_columns(fn):
    result=fachler(facts())
    mapping={'period':'date','segment':'sector','allocation':'alloc','selection':'select','interaction':'cross',
        'portfolio_return':'pr','benchmark_return':'br','excess_return':'active'}
    detail=result.detail.rename({k:v for k,v in mapping.items() if k in result.detail.columns})
    summary=result.summary.rename({k:v for k,v in mapping.items() if k in result.summary.columns})
    opts={'column_'+k:v for k,v in mapping.items()}
    actual=fn(AttributionResult(detail,summary),**opts)
    expected=fn(result)
    assert_frame_equal(actual.summary,expected.summary)


def test_currency_premium_independent_columns():
    market=pl.DataFrame({'period':[1,1], 'segment':['a','b'], 'portfolio_weight':[.6,.4], 'benchmark_weight':[.5,.5],
        'portfolio_return':[.08,.02], 'benchmark_return':[.05,.03], 'local_cash_return':[0.,0.]})
    currency=market.drop('local_cash_return')
    expected=currency_premiums(market,currency)
    m=market.rename({'period':'date','segment':'market','portfolio_weight':'pm','benchmark_weight':'bm','portfolio_return':'mrp','benchmark_return':'mrb','local_cash_return':'cash'})
    c=currency.rename({'period':'date','segment':'currency','portfolio_weight':'pc','benchmark_weight':'bc','portfolio_return':'crp','benchmark_return':'crb'})
    actual=currency_premiums(m,c,column_period='date',column_local_cash_return='cash',
        market_columns={'segment':'market','portfolio_weight':'pm','benchmark_weight':'bm','portfolio_return':'mrp','benchmark_return':'mrb'},
        currency_columns={'segment':'currency','portfolio_weight':'pc','benchmark_weight':'bc','portfolio_return':'crp','benchmark_return':'crb'})
    market_inverse={'date':'period','market':'segment','pm':'portfolio_weight','bm':'benchmark_weight',
        'mrp':'portfolio_return','mrb':'benchmark_return','cash':'local_cash_return'}
    currency_inverse={'date':'period','currency':'segment','pc':'portfolio_weight','bc':'benchmark_weight',
        'crp':'portfolio_return','crb':'benchmark_return'}
    assert set(market_inverse).issubset(actual[0].columns)
    assert set(currency_inverse).issubset(actual[1].columns)
    assert_frame_equal(actual[0].rename(market_inverse),expected[0])
    assert_frame_equal(actual[1].rename(currency_inverse),expected[1])
    assert_frame_equal(actual[2].rename({'date':'period'}),expected[2])
    md,cd,sm=expected
    ren={'period':'date','segment':'group','market_allocation':'ma','security_selection':'ss','currency_allocation':'ca','hedge_selection':'hs',
         'portfolio_log_return':'plr','benchmark_log_return':'blr','active_log_return':'alr'}
    def renamed(frame): return frame.rename({k:v for k,v in ren.items() if k in frame.columns})
    opts={'column_'+k:v for k,v in ren.items()}
    outputs=roll_up_currency_premiums(renamed(md),renamed(cd),renamed(sm),**opts)
    for a,b in zip(outputs,roll_up_currency_premiums(md,cd,sm)):
        assert_frame_equal(a.rename({v:k for k,v in ren.items() if v in a.columns}),b)


def test_invalid_mappings_rejected():
    with pytest.raises(AttributionError,match='nonempty strings'): fachler(facts(),column_segment='')
    with pytest.raises(AttributionError,match='missing columns'): fachler(facts(),column_segment='missing')
    both=fachler(facts().with_columns(pl.col('segment').alias('sector')),column_segment='sector')
    assert {'segment','sector'} <= set(both.detail.columns)
    with pytest.raises(AttributionError,match='conflict'): fachler(facts().with_columns(pl.lit(1).alias('allocation')))
    # Simultaneous swaps are allowed and do not overwrite data.
    expected=fachler(facts())
    swapped=facts().rename({'portfolio_return':'benchmark_return','benchmark_return':'portfolio_return'})
    actual=fachler(swapped,column_portfolio_return='benchmark_return',column_benchmark_return='portfolio_return')
    assert_frame_equal(actual.summary,expected.summary)


def test_one_return_column_can_supply_both_roles():
    data=pl.DataFrame({'period':[1,1], 'segment':['a','b'], 'portfolio_weight':[.6,.4],
        'benchmark_weight':[.5,.5], 'Return':[.05,.01]})
    result=fachler(data,column_portfolio_return='Return',column_benchmark_return='Return')
    assert result.summary['selection'][0] == pytest.approx(0.)
    assert result.summary['excess_return'][0] == pytest.approx(.004)


def covariance(keys):
    return pl.DataFrame([{'row':a,'column':b,'covariance':float(a==b)} for a in keys for b in keys])


def test_hentschel_factor_and_covariances():
    data=pl.DataFrame({'asset':['x','y','z'], 'return':[.03,.01,.08], 'portfolio_weight':[.7,.2,.1],
       'benchmark_weight':[.2,.3,.5], 'intercept':[1.,1.,1.], 'value':[-1.,0.,1.], 'regression_weight':[1.,2.,1.]})
    remap={'asset':'id','return':'ret','portfolio_weight':'pw','benchmark_weight':'bw','regression_weight':'reg_w'}
    actual=factor_attribution_restricted(data.rename(remap),['intercept','value'],column_asset='id',column_return='ret',
        column_portfolio_weight='pw',column_benchmark_weight='bw',column_regression_weight='reg_w')
    assert_frame_equal(actual.sort('factor'),factor_attribution_restricted(data,['intercept','value']).sort('factor'))
    with pytest.raises(AttributionError,match='missing columns'):
        factor_attribution_restricted(data,['intercept','value'],column_regression_weight='missing')
    stocks=pl.DataFrame({'asset':['a','b','c','d'],'segment':['x','x','y','y'],
        'portfolio_weight':[.4,.2,.1,.3],'benchmark_weight':[.25,.25,.25,.25]})
    c=covariance(['a','b','c','d'])
    cr=c.rename({'row':'r','column':'c','covariance':'v'})
    kwargs={'column_asset':'id','column_segment':'group','column_benchmark_weight':'bw',
        'column_covariance_row':'r','column_covariance_column':'c','column_covariance_value':'v'}
    custom=stocks.rename({'asset':'id','segment':'group','portfolio_weight':'pw','benchmark_weight':'bw'})
    assert_frame_equal(benchmark_residual_covariance(custom,cr,**kwargs),benchmark_residual_covariance(stocks,c))
    assert_frame_equal(brinson_group_covariance(custom,cr,column_portfolio_weight='pw',**kwargs),brinson_group_covariance(stocks,c))


@pytest.mark.parametrize('fn',[brinson_interaction_restricted,brinson_interaction_reconciled])
def test_hentschel_brinson_custom_columns(fn):
    frame=facts().filter(pl.col('period')==1)
    c=covariance(['B:a','B:b','P:a','P:b'])
    opts={k:v for k,v in ARGS.items() if k not in ('column_local_return','column_currency_return')}
    actual=fn(frame.rename({k:v for k,v in MAPPING.items() if k not in ('local_return','currency_return')}),
       c.rename({'row':'r','column':'c','covariance':'v'}),column_covariance_row='r',column_covariance_column='c',column_covariance_value='v',**opts)
    assert 'date' in actual.summary.columns
    assert_frame_equal(actual.summary,fn(frame,c).summary.rename({'period':'date'}))


def test_hentschel_link_custom_columns():
    result=fachler(facts())
    mapping={'period':'date','segment':'sector','allocation':'alloc','selection':'select','interaction':'cross',
        'portfolio_return':'pr','benchmark_return':'br','excess_return':'active'}
    modified=AttributionResult(result.detail.rename({k:v for k,v in mapping.items() if k in result.detail.columns}),
        result.summary.rename({k:v for k,v in mapping.items() if k in result.summary.columns}))
    opts={'column_'+k:v for k,v in mapping.items()}
    c=covariance([1,2])
    cr=c.rename({'row':'r','column':'c','covariance':'v'})
    baseline=pl.DataFrame({'date':[1,2],'base':[1.,1.]})
    actual=link_precision_weighted(modified,cr,baseline,**opts,column_covariance_row='r',column_covariance_column='c',
        column_covariance_value='v',column_baseline_weight='base')
    assert_frame_equal(actual.summary,link_precision_weighted(result,c).summary)
    entries=covariance(list(range(12)))
    actual=link_entry_reconciled(modified,entries.rename({'row':'r','column':'c','covariance':'v'}),**opts,
        column_covariance_row='r',column_covariance_column='c',column_covariance_value='v')
    assert_frame_equal(actual.summary,link_entry_reconciled(result,entries).summary)


def test_column_names_preserved_through_calculation_linking_and_reporting(capsys):
    from brinsons import calculate_multiple_period, print_multiple_period_category_weights, chart_multiple_period_excess_return_waterfall
    import matplotlib.pyplot as plt
    data=facts().select(pl.col('period').alias('When'),pl.col('segment').alias('Sector'),
        pl.col('portfolio_weight').alias('P_Weight'),pl.col('benchmark_weight').alias('B_Weight'),
        pl.col('portfolio_return').alias('Return'))
    schema=ColumnNames(period='When',segment='Sector',portfolio_weight='P_Weight',
        benchmark_weight='B_Weight',portfolio_return='Return',benchmark_return='Return')
    original=data.clone()
    direct=fachler(data,columns=schema)
    assert direct.columns == schema
    assert set(data.columns).issubset(direct.detail.columns)
    assert 'When' in direct.summary.columns and 'period' not in direct.summary.columns
    linked=link_menchero(direct)
    assert linked.columns.period=='When' and linked.columns.segment=='Sector'
    assert 'When' in linked.detail.columns and 'Sector' in linked.detail.columns
    workflow=calculate_multiple_period(data,columns=schema)
    assert_frame_equal(workflow.linked.summary,linked.summary)
    assert 'Category Weights' in print_multiple_period_category_weights(direct)
    fig,ax=chart_multiple_period_excess_return_waterfall(direct)
    assert len(ax.patches)==7
    plt.close(fig)
    assert_frame_equal(data,original)


def test_column_names_class_controls_effect_and_summary_fields():
    frame=facts().select(pl.col('period').alias('When'),pl.col('segment').alias('Sector'),
        pl.col('portfolio_weight').alias('PW'),pl.col('benchmark_weight').alias('BW'),
        pl.col('portfolio_return').alias('RP'),pl.col('benchmark_return').alias('RB'))
    names=ColumnNames(period='When',segment='Sector',portfolio_weight='PW',benchmark_weight='BW',
        portfolio_return='RP',benchmark_return='RB',allocation='A',selection='S',interaction='I',
        summary_allocation='TotalA',summary_selection='TotalS',summary_interaction='TotalI',
        summary_portfolio_return='PortfolioTotal',summary_benchmark_return='BenchmarkTotal',
        summary_excess_return='ActiveTotal')
    result=fachler(frame,columns=names)
    assert set(names.effects).issubset(result.detail.columns)
    assert set(names.summary_effects).issubset(result.summary.columns)
    linked=link_menchero(result)
    assert linked.columns.period=='When'
    assert 'When' in linked.detail.columns
    assert abs(linked.summary['allocation'][0]+linked.summary['selection'][0]+
               linked.summary['interaction'][0]-linked.summary['excess_return'][0]) < 1e-10
