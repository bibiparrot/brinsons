import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polars as pl
import pytest
from brinsons import (AttributionError, AttributionResult, fachler, link_menchero,
    print_category_weights, print_category_returns, print_category_attribution,
    print_total_attribution, print_attribution_summary,
    chart_category_returns, chart_category_weights, chart_category_attribution)


def data():
    return pl.DataFrame({'period':[1,1,2,2], 'segment':['A','B','A','B'],
        'portfolio_weight':[.6,.4,.55,.45], 'benchmark_weight':[.5,.5,.5,.5],
        'portfolio_return':[.12,.04,.03,.01], 'benchmark_return':[.10,.06,.02,.015]})


def test_markdown_reports_close_and_show_periods(capsys):
    r=fachler(data())
    weights=print_category_weights(r)
    assert '| A | 57.5000% | 50.0000% | 7.5000% |' in weights
    assert 'Category Weights' in capsys.readouterr().out
    returns=print_category_returns(r)
    assert '| A | 15.3600% | 12.2000% |' in returns
    assert 'Period 1' in print_category_weights(r,period=1)
    total=link_menchero(r).summary.row(0,named=True)
    total_text=print_total_attribution(r)
    assert f"{total['excess_return']*100:.4f}%" in total_text
    category=print_category_attribution(r)
    assert all(x in category for x in ['Allocation','Selection','Interaction','Active'])
    summary=print_attribution_summary(r)
    assert f"{total['portfolio_return']*100:.4f}%" in summary
    with pytest.raises(AttributionError,match='not present'):
        print_category_weights(r,period=999)


def test_multiple_period_outputs_identify_linking_method():
    from brinsons import (
        print_multiple_period_category_weights, print_multiple_period_category_returns,
        print_multiple_period_category_attribution, print_multiple_period_total_attribution,
        print_multiple_period_attribution_summary, chart_multiple_period_category_weights,
        chart_multiple_period_category_returns, chart_multiple_period_category_attribution,
        chart_multiple_period_excess_return_waterfall,
    )
    result = fachler(data())
    for report in (print_multiple_period_category_weights, print_multiple_period_category_returns):
        assert 'Linking: none' in report(result)
    for report in (print_multiple_period_category_attribution, print_multiple_period_total_attribution,
                   print_multiple_period_attribution_summary):
        assert 'Linking: Menchero' in report(result)
    for chart in (chart_multiple_period_category_weights, chart_multiple_period_category_returns):
        fig, ax = chart(result)
        assert 'Linking: none' in ax.get_title()
        plt.close(fig)
    fig, ax = chart_multiple_period_category_attribution(result)
    assert 'Linking: Menchero' in ax.get_title()
    plt.close(fig)
    fig, ax = chart_multiple_period_excess_return_waterfall(result)
    assert 'Linking: Menchero' in ax.get_title(loc='left')
    plt.close(fig)


def test_multiple_period_reporting_accepts_explicit_linkers():
    from brinsons import (
        link_carino, link_grap, link_frongello, link_geometric,
        print_multiple_period_category_attribution, print_multiple_period_total_attribution,
        print_multiple_period_attribution_summary, chart_multiple_period_category_attribution,
        chart_multiple_period_excess_return_waterfall,
    )
    result = fachler(data())
    for linking, label in ((link_carino, 'Cariño'), (link_grap, 'GRAP'),
                           (link_frongello, 'Frongello'), (link_menchero, 'Menchero')):
        expected = linking(result).summary['allocation'][0]
        for report in (print_multiple_period_category_attribution, print_multiple_period_total_attribution,
                       print_multiple_period_attribution_summary):
            output = report(result, linking=linking)
            assert f'Linking: {label}' in output
        assert f'{expected*100:.4f}%' in print_multiple_period_total_attribution(result, linking=linking)
        for chart in (chart_multiple_period_category_attribution, chart_multiple_period_excess_return_waterfall):
            fig, ax = chart(result, linking=linking)
            assert f'Linking: {label}' in (ax.get_title() + ax.get_title(loc='left'))
            plt.close(fig)
    geometric = print_multiple_period_total_attribution(result, linking=link_geometric)
    assert 'Linking: Geometric' in geometric and 'Cross term' in geometric
    assert 'Relative active return' in print_multiple_period_attribution_summary(result, linking=link_geometric)
    with pytest.raises(AttributionError, match='segment-level arithmetic'):
        chart_multiple_period_category_attribution(result, linking=link_geometric)
    with pytest.raises(TypeError, match='linking must be'):
        print_multiple_period_total_attribution(result, linking='Carino')


@pytest.mark.parametrize('fn,n',[(chart_category_returns,4),(chart_category_weights,6),(chart_category_attribution,6)])
def test_chart_values_and_period(fn,n):
    r=fachler(data())
    fig,ax=fn(r)
    assert len(ax.patches)==n
    assert len(ax.get_yticklabels())==2
    plt.close(fig)
    fig,ax=fn(r,period=1)
    assert len(ax.patches)==n
    if fn is chart_category_returns:
        assert ax.patches[0].get_width()==pytest.approx(.12)
        assert ax.patches[2].get_width()==pytest.approx(.10)
    if fn is chart_category_attribution:
        effect=r.detail.filter(pl.col('period')==1)['allocation'][0]
        assert ax.patches[0].get_width()==pytest.approx(effect)
    plt.close(fig)


def test_single_period_and_custom_result_names(capsys):
    r=fachler(data().filter(pl.col('period')==1))
    assert 'All periods' in print_total_attribution(r)
    mapping={'period':'t','segment':'g','portfolio_weight':'wp','benchmark_weight':'wb',
        'portfolio_return':'rp','benchmark_return':'rb','excess_return':'active',
        'allocation':'a','selection':'s','interaction':'i'}
    rename=lambda frame: frame.rename({k:v for k,v in mapping.items() if k in frame.columns})
    custom=AttributionResult(rename(r.detail),rename(r.summary))
    opts={'column_'+k:v for k,v in mapping.items()}
    assert print_category_weights(custom,**opts)==print_category_weights(r)
    fig,ax=chart_category_attribution(custom,**opts)
    assert len(ax.patches)==6
    plt.close(fig)


def test_categories_absent_in_some_periods(capsys):
    df=pl.DataFrame({'period':[1,1,2], 'segment':['A','B','A'],
        'portfolio_weight':[.6,.4,1.], 'benchmark_weight':[.5,.5,1.],
        'portfolio_return':[.12,.04,.03], 'benchmark_return':[.10,.06,.02]})
    r=fachler(df)
    weights=print_category_weights(r)
    assert '| B | 20.0000% | 25.0000% | -5.0000% |' in weights
    assert '| B |' in print_category_attribution(r)
    with pytest.raises(AttributionError,match='every period'):
        print_category_returns(r)


@pytest.mark.parametrize('single,multiple,original', [
    ('print_single_period_category_weights','print_multiple_period_category_weights','print_category_weights'),
    ('print_single_period_category_returns','print_multiple_period_category_returns','print_category_returns'),
    ('print_single_period_category_attribution','print_multiple_period_category_attribution','print_category_attribution'),
    ('print_single_period_total_attribution','print_multiple_period_total_attribution','print_total_attribution'),
    ('print_single_period_attribution_summary','print_multiple_period_attribution_summary','print_attribution_summary'),
])
def test_explicit_period_markdown(single,multiple,original,capsys):
    import brinsons as b
    r=fachler(data())
    assert getattr(b,single)(r,1) == getattr(b,original)(r,period=1)
    assert getattr(b,multiple)(r) == getattr(b,original)(r)
    with pytest.raises(TypeError): getattr(b,single)(r)
    with pytest.raises(AttributionError,match='requires a period'): getattr(b,single)(r,None)
    with pytest.raises(AttributionError,match='at least two'): getattr(b,multiple)(fachler(data().filter(pl.col('period')==1)))


@pytest.mark.parametrize('single,multiple,original', [
    ('chart_single_period_category_returns','chart_multiple_period_category_returns','chart_category_returns'),
    ('chart_single_period_category_weights','chart_multiple_period_category_weights','chart_category_weights'),
    ('chart_single_period_category_attribution','chart_multiple_period_category_attribution','chart_category_attribution'),
])
def test_explicit_period_charts(single,multiple,original):
    import brinsons as b
    r=fachler(data())
    for f,args,old_args in [(getattr(b,single),(r,1),{'period':1}),(getattr(b,multiple),(r,),{})]:
        fig,ax=f(*args)
        old_fig,old_ax=getattr(b,original)(r,**old_args)
        assert [(p.get_width(),p.get_x()) for p in ax.patches] == pytest.approx([(p.get_width(),p.get_x()) for p in old_ax.patches])
        plt.close(fig); plt.close(old_fig)
    with pytest.raises(AttributionError,match='at least two'): getattr(b,multiple)(fachler(data().filter(pl.col('period')==1)))


def test_explicit_period_custom_columns():
    from brinsons import print_multiple_period_category_weights, chart_single_period_category_returns
    r=fachler(data())
    mapping={'period':'t','segment':'g','portfolio_weight':'wp','benchmark_weight':'wb',
        'portfolio_return':'rp','benchmark_return':'rb','excess_return':'active',
        'allocation':'a','selection':'s','interaction':'i'}
    rename=lambda frame: frame.rename({k:v for k,v in mapping.items() if k in frame.columns})
    custom=AttributionResult(rename(r.detail),rename(r.summary))
    opts={'column_'+k:v for k,v in mapping.items()}
    assert print_multiple_period_category_weights(custom,**opts) == print_multiple_period_category_weights(r)
    fig,ax=chart_single_period_category_returns(custom,2,**opts)
    assert ax.patches[0].get_width()==pytest.approx(.03)
    plt.close(fig)


@pytest.mark.parametrize('by_category',[True,False])
def test_single_period_waterfall_reconciles(by_category):
    from brinsons import chart_single_period_excess_return_waterfall
    result=fachler(data())
    fig,ax=chart_single_period_excess_return_waterfall(result,1,by_category=by_category)
    detail=result.detail.filter(pl.col('period')==1).sort('segment')
    values=[]
    for name in ['allocation','selection','interaction']:
        values.extend(detail[name].to_list() if by_category else [detail[name].sum()])
    assert len(ax.patches)==len(values)+1
    running=0.0
    for bar,value in zip(ax.patches,values,strict=False):
        assert bar.get_y()==pytest.approx(min(running,running+value*10000))
        assert bar.get_height()==pytest.approx(abs(value*10000))
        running+=value*10000
    total=result.summary.filter(pl.col('period')==1)['excess_return'][0]*10000
    assert running==pytest.approx(total)
    assert ax.patches[-1].get_y()==pytest.approx(min(total,0))
    assert ax.patches[-1].get_height()==pytest.approx(abs(total))
    assert 'basis points' in ax.get_ylabel()
    assert [label.get_text() for label in ax.get_xticklabels()][-1]=='Total active return'
    low,high=ax.get_ylim()
    assert low < min(bar.get_y() for bar in ax.patches)
    assert high > max(bar.get_y()+bar.get_height() for bar in ax.patches)
    plt.close(fig)


def test_multiple_period_waterfall_and_guards():
    from brinsons import chart_multiple_period_excess_return_waterfall,chart_single_period_excess_return_waterfall
    result=fachler(data())
    fig,ax=chart_multiple_period_excess_return_waterfall(result,by_category=False)
    linked=link_menchero(result).summary.row(0,named=True)
    assert len(ax.patches)==4
    for bar,name in zip(ax.patches,['allocation','selection','interaction','excess_return']):
        assert bar.get_height()==pytest.approx(abs(linked[name]*10000))
    assert 'Menchero' in ax.get_title(loc='left')
    plt.close(fig)
    with pytest.raises(AttributionError,match='at least two'):
        chart_multiple_period_excess_return_waterfall(fachler(data().filter(pl.col('period')==1)))
    with pytest.raises(AttributionError,match='not present'):
        chart_single_period_excess_return_waterfall(result,999)
    with pytest.raises(TypeError,match='by_category'):
        chart_single_period_excess_return_waterfall(result,1,by_category='yes')


def test_multiple_period_waterfall_all_linkers():
    from brinsons import (chart_multiple_period_excess_return_waterfall, link_carino,
                          link_grap, link_frongello, link_geometric)
    result = fachler(data())
    for linking, name in ((link_carino, 'Cariño'), (link_grap, 'GRAP'),
                          (link_frongello, 'Frongello'), (link_menchero, 'Menchero'),
                          (link_geometric, 'Geometric')):
        fig, ax = chart_multiple_period_excess_return_waterfall(result, linking=linking,
            **({} if linking is link_geometric else {'by_category': False}))
        summary = linking(result).summary.row(0, named=True)
        pieces = [summary['allocation'], summary['selection']]
        if linking is link_geometric:
            pieces.append(summary['allocation'] * summary['selection'])
            assert [label.get_text() for label in ax.get_xticklabels()][-1] == 'Relative active return'
        else:
            pieces.append(summary['interaction'])
            assert [label.get_text() for label in ax.get_xticklabels()][-1] == 'Total active return'
        assert sum(pieces) == pytest.approx(summary['excess_return'], abs=1e-10)
        assert [bar.get_height() for bar in ax.patches] == pytest.approx(
            [abs(value * 10000) for value in pieces + [summary['excess_return']]])
        assert name in ax.get_title(loc='left')
        plt.close(fig)
    with pytest.raises(AttributionError, match='by_category=False'):
        chart_multiple_period_excess_return_waterfall(result, linking=link_geometric, by_category=True)


def test_waterfall_negative_and_zero_totals():
    from brinsons import chart_single_period_excess_return_waterfall
    frame=data().filter(pl.col('period')==1)
    for returns in ([0.01,0.01],[.04,.04]):
        r=fachler(frame.with_columns(pl.Series('portfolio_return',returns)))
        fig,ax=chart_single_period_excess_return_waterfall(r,1,by_category=False)
        total=r.summary['excess_return'][0]*10000
        assert total <= 1e-10
        assert ax.patches[-1].get_y()==pytest.approx(min(0,total))
        assert ax.patches[-1].get_height()==pytest.approx(abs(total))
        plt.close(fig)


def test_waterfall_custom_column_names():
    from brinsons import chart_multiple_period_excess_return_waterfall
    r=fachler(data())
    mapping={'period':'t','segment':'g','portfolio_weight':'wp','benchmark_weight':'wb',
        'portfolio_return':'rp','benchmark_return':'rb','excess_return':'active',
        'allocation':'a','selection':'s','interaction':'i'}
    rename=lambda frame: frame.rename({k:v for k,v in mapping.items() if k in frame.columns})
    custom=AttributionResult(rename(r.detail),rename(r.summary))
    fig,ax=chart_multiple_period_excess_return_waterfall(custom,**{'column_'+k:v for k,v in mapping.items()})
    assert ax.patches[-1].get_height()==pytest.approx(abs(link_menchero(r).summary['excess_return'][0]*10000))
    plt.close(fig)
