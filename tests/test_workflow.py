import polars as pl
import pytest
from polars.testing import assert_frame_equal
from brinsons import (AttributionError, AttributionResult, MultiplePeriodAttribution, bhb, fachler,
    link_carino, link_grap, link_frongello, link_menchero, link_geometric,
    calculate_single_period, calculate_multiple_period)


def facts():
    return pl.DataFrame({'period':[1,1,2,2,3,3], 'segment':['A','B','A','B','A','B'],
        'portfolio_weight':[.6,.4,.55,.45,.48,.52], 'benchmark_weight':[.5]*6,
        'portfolio_return':[.12,.04,.03,.01,-.02,.01], 'benchmark_return':[.1,.06,.02,.015,-.01,.02]})


@pytest.mark.parametrize('attribution',[bhb,fachler])
@pytest.mark.parametrize('effects',[2,3])
def test_single_period_method_and_selection(attribution,effects):
    df=facts()
    actual=calculate_single_period(df,period=2,attribution=attribution,effects=effects)
    expected=attribution(df.filter(pl.col('period')==2),effects=effects)
    assert_frame_equal(actual.detail,expected.detail)
    assert_frame_equal(actual.summary,expected.summary)
    assert_frame_equal(calculate_single_period(df.filter(pl.col('period')==2),attribution=attribution,effects=effects).summary,expected.summary)
    assert df.height==6


@pytest.mark.parametrize('attribution',[bhb,fachler])
@pytest.mark.parametrize('linking',[link_carino,link_grap,link_frongello,link_menchero,link_geometric])
def test_multiple_period_method_selection(attribution,linking):
    df=facts()
    actual=calculate_multiple_period(df,attribution=attribution,linking=linking)
    periods=attribution(df)
    expected=linking(periods)
    assert isinstance(actual,MultiplePeriodAttribution)
    assert_frame_equal(actual.periods.detail,periods.detail)
    assert_frame_equal(actual.periods.summary,periods.summary)
    assert_frame_equal(actual.linked.detail,expected.detail)
    assert_frame_equal(actual.linked.summary,expected.summary)


def test_custom_columns_and_shared_return():
    df=facts().rename({'period':'Period','segment':'Category','portfolio_weight':'PortfolioWeight',
        'benchmark_weight':'BenchmarkWeight','portfolio_return':'Return'}).drop('benchmark_return')
    opts=dict(column_period='Period',column_segment='Category',column_portfolio_weight='PortfolioWeight',
        column_benchmark_weight='BenchmarkWeight',column_portfolio_return='Return',column_benchmark_return='Return')
    single=calculate_single_period(df,period=1,**opts)
    multi=calculate_multiple_period(df,**opts)
    assert single.summary['selection'][0]==pytest.approx(0)
    assert single.summary['Period'][0]==1
    assert multi.periods.summary.height==3
    assert multi.linked.summary.height==1


def test_period_and_callable_validation():
    df=facts()
    with pytest.raises(AttributionError,match='exactly one period'): calculate_single_period(df)
    with pytest.raises(AttributionError,match='not present'): calculate_single_period(df,period=999)
    with pytest.raises(AttributionError,match='at least two periods'): calculate_multiple_period(df.filter(pl.col('period')==1))
    with pytest.raises(TypeError,match='callable'): calculate_single_period(df,period=1,attribution='bhb')
    with pytest.raises(TypeError,match='callable'): calculate_multiple_period(df,linking='carino')
    with pytest.raises(AttributionError,match='nonempty'): calculate_single_period(df,column_period='')
    with pytest.raises(AttributionError,match='one-period'): calculate_single_period(df,period=1,attribution=lambda *a,**k: None)
    with pytest.raises(AttributionError,match='single-horizon'): calculate_multiple_period(df,linking=lambda r: None)


def test_default_choices_and_two_effect_multiple_period():
    df=facts()
    actual=calculate_multiple_period(df,effects=2)
    expected=link_menchero(fachler(df,effects=2))
    assert_frame_equal(actual.linked.summary,expected.summary)
    assert actual.periods.summary['interaction'].to_list()==pytest.approx([0.,0.,0.])
