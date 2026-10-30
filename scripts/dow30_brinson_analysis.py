"""Reproduce a MathWorks-style monthly Brinson analysis with Yahoo prices.

Run: python scripts/dow30_brinson_analysis.py
Requires yfinance, matplotlib, Polars, and the local brinsons package.
The cached metadata from scripts/fetch_dow30.py supplies the present-day sectors.
"""
from datetime import date
from pathlib import Path
import json
import polars as pl
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from brinsons import fachler, link_menchero

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dow30_brinson'
OUT = ROOT / 'outputs' / 'dow30_brinson'
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
source = json.loads((ROOT/'data'/'dow30_yfinance_data.json').read_text(encoding='utf-8'))
tickers = sorted(source['tickers'])
sectors = {r[0]:r[2] for r in source['metadata']}
if len(tickers) != 30 or len(set(tickers)) != 30 or any(not sectors.get(t) for t in tickers):
    raise ValueError('Expected 30 unique constituents with known sectors')
# Include a prior month-end price for 12 complete return months.
today = date.today()
end = today.replace(day=1)
month_index = end.year*12 + end.month-1 - 13
start = date(month_index//12, month_index%12+1, 1)
raw = yf.download(tickers, start=start.isoformat(), end=end.isoformat(), interval='1d',
                  auto_adjust=True, group_by='ticker', threads=True, progress=False, timeout=25)
rows=[]
for ticker in tickers:
    if ticker not in raw.columns.get_level_values(0):
        raise ValueError(f'Missing price data: {ticker}')
    for day, val in raw[ticker]['Close'].dropna().items():
        rows.append({'date':day.date(), 'ticker':ticker, 'close':float(val)})
prices=pl.DataFrame(rows).sort(['ticker','date']).with_columns(pl.col('date').dt.strftime('%Y-%m').alias('month'))
monthly=prices.group_by(['ticker','month']).agg(pl.col('date').last().alias('date'),pl.col('close').last().alias('close')).sort(['ticker','month'])
months=sorted(monthly['month'].unique().to_list())[-13:]
if len(months) != 13:
    raise ValueError(f'Expected 13 monthly price observations, got {len(months)}')
monthly=monthly.filter(pl.col('month').is_in(months)).with_columns(
    pl.col('close').shift(1).over('ticker').alias('previous_close'),
    pl.col('date').shift(1).over('ticker').alias('previous_date'))
monthly=monthly.filter(pl.col('previous_close').is_not_null()).with_columns(
    (pl.col('close')/pl.col('previous_close')-1).alias('Return'))
if monthly.height != 360 or monthly.select(pl.struct('ticker','month').n_unique()).item() != 360:
    raise ValueError('Missing or duplicate monthly observations')
periods=monthly.group_by('month').agg(pl.col('date').n_unique().alias('n_end'),pl.col('previous_date').n_unique().alias('n_start'),
    pl.col('date').first().alias('PeriodEnd'),pl.col('previous_date').first().alias('PeriodStart')).sort('month').with_row_index('Period',offset=1)
if periods.filter((pl.col('n_end')!=1)|(pl.col('n_start')!=1)).height:
    raise ValueError('Month-end observations do not align for every stock')
# Factors use only ticker, sector, and quarter index, not returns.
sector_tilt={'Technology':1.35,'Healthcare':1.12,'Financial Services':1.12,
 'Consumer Cyclical':1.07,'Communication Services':1.05,'Industrials':1.0,
 'Consumer Defensive':.90,'Energy':.82,'Basic Materials':.85}
def weight_factor(ticker, p):
    i=tickers.index(ticker)
    quarter=(p-1)//3
    return sector_tilt.get(sectors[ticker],1.0)*(0.80+0.08*((i+quarter*2)%6))
rows=[]
for r in monthly.join(periods.select('month','Period'),on='month').iter_rows(named=True):
    t=r['ticker']; p=r['Period']
    rows.append({'Period':p,'Name':t,'Return':r['Return'],'Category':sectors[t],
                 '_raw_weight':weight_factor(t,p),'BenchmarkWeight':1.0/30})
asset=pl.DataFrame(rows).with_columns((pl.col('_raw_weight')/pl.col('_raw_weight').sum().over('Period')).alias('PortfolioWeight')).select(
 'Period','Name','Return','Category','PortfolioWeight','BenchmarkWeight').sort(['Period','Name'])
weights=asset.group_by('Period').agg(pl.col('PortfolioWeight','BenchmarkWeight').sum())
assert weights.select(((pl.col('PortfolioWeight')-1).abs()<1e-12).all()).item()
assert weights.select(((pl.col('BenchmarkWeight')-1).abs()<1e-12).all()).item()
category=asset.group_by(['Period','Category']).agg(
    pl.col('PortfolioWeight').sum().alias('portfolio_weight'), pl.col('BenchmarkWeight').sum().alias('benchmark_weight'),
    (pl.col('PortfolioWeight')*pl.col('Return')).sum().alias('p_contribution'),
    (pl.col('BenchmarkWeight')*pl.col('Return')).sum().alias('b_contribution')).with_columns(
    (pl.col('p_contribution')/pl.col('portfolio_weight')).alias('portfolio_return'),
    (pl.col('b_contribution')/pl.col('benchmark_weight')).alias('benchmark_return')).rename({'Period':'period','Category':'segment'}).sort(['period','segment'])
result=fachler(category.select('period','segment','portfolio_weight','benchmark_weight','portfolio_return','benchmark_return'))
linked=link_menchero(result)
period_summary=result.summary.sort('period')
direct=asset.group_by('Period').agg((pl.col('PortfolioWeight')*pl.col('Return')).sum().alias('p'),
 (pl.col('BenchmarkWeight')*pl.col('Return')).sum().alias('b')).sort('Period')
for a,b in zip(direct.iter_rows(named=True),period_summary.iter_rows(named=True)):
    assert abs(a['p']-b['portfolio_return'])<1e-12 and abs(a['b']-b['benchmark_return'])<1e-12
s=linked.summary.row(0,named=True)
assert abs(sum(s[x] for x in ['allocation','selection','interaction'])-s['excess_return']) < 1e-12
agg=linked.detail.group_by('segment').agg(pl.col('linked_allocation','linked_selection','linked_interaction').sum()).with_columns(
 pl.sum_horizontal('linked_allocation','linked_selection','linked_interaction').alias('active_return'))
category_agg=category.group_by('segment').agg(pl.col('portfolio_weight','benchmark_weight').mean(),
 ((1+pl.col('portfolio_return')).product()-1).alias('portfolio_category_return'),
 ((1+pl.col('benchmark_return')).product()-1).alias('benchmark_category_return')).with_columns(
 (pl.col('portfolio_weight')-pl.col('benchmark_weight')).alias('active_weight')).join(agg,on='segment').sort('segment')
category_period=linked.detail.join(category.select('period','segment','portfolio_weight','benchmark_weight','portfolio_return','benchmark_return'),on=['period','segment']).select(
 'period','segment','portfolio_weight','benchmark_weight','portfolio_return','benchmark_return','allocation','selection','interaction','factor','linked_allocation','linked_selection','linked_interaction').sort(['period','segment'])
period_summary=period_summary.join(linked.detail.group_by('period').agg(pl.col('factor').first()),on='period').join(periods.select('Period','PeriodStart','PeriodEnd'),left_on='period',right_on='Period').sort('period')
for name,df in [('asset_table',asset),('periods',periods.select('Period','PeriodStart','PeriodEnd')),
                ('monthly_prices',monthly.select('month','date','ticker','close','previous_close')),
                ('category_summary',category_agg),('category_period',category_period),('period_summary',period_summary)]:
    df.write_csv(DATA/f'{name}.csv')
summary={'portfolio_return':s['portfolio_return'],'benchmark_return':s['benchmark_return'],'active_return':s['excess_return'],
 'allocation':s['allocation'],'selection':s['selection'],'interaction':s['interaction'],'n_periods':len(months)-1,
 'linking_method':'Menchero',
 'n_assets':len(tickers),'n_categories':category_agg.height,'first_start':str(periods['PeriodStart'][0]),'last_end':str(periods['PeriodEnd'][-1]),
 'metadata_source_retrieved_at_utc':source['retrieved_at_utc'],'yfinance_version':yf.__version__}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

# Charts follow the three comparisons in the MathWorks example.
labels=category_agg['segment'].to_list(); y=list(range(len(labels))); width=.36
fig,axes=plt.subplots(1,3,figsize=(20,7.5),layout='constrained')
fig.suptitle('Dow 30 example: sector attribution — Linking: Menchero',fontsize=16,fontweight='bold')
ax=axes[0]; ax.barh([i-width/2 for i in y],category_agg['portfolio_category_return'].to_list(),height=width,label='Portfolio',color='#2366a8'); ax.barh([i+width/2 for i in y],category_agg['benchmark_category_return'].to_list(),height=width,label='Benchmark',color='#aab5c5'); ax.set_title('Compounded sector returns — Linking: none'); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.axvline(0,color='#5c6570',lw=.6); ax.legend(fontsize=8)
ax=axes[1]; ax.barh([i-width/2 for i in y],category_agg['portfolio_weight'].to_list(),height=width,label='Portfolio',color='#2366a8'); ax.barh([i+width/2 for i in y],category_agg['benchmark_weight'].to_list(),height=width,label='Benchmark',color='#aab5c5'); ax.set_title('Average monthly sector weights — Linking: none'); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.legend(fontsize=8)
ax=axes[2]; pos=[0.]*len(y); neg=[0.]*len(y)
for col,label,color in [('linked_allocation','Allocation','#2366a8'),('linked_selection','Selection','#3f8d70'),('linked_interaction','Interaction','#df9d50')]:
    vals=category_agg[col].to_list(); left=[pos[i] if v>=0 else neg[i] for i,v in enumerate(vals)]
    ax.barh(y,vals,left=left,height=.65,label=label,color=color)
    for i,v in enumerate(vals):
        if v>=0: pos[i]+=v
        else: neg[i]+=v
ax.set_title('Linked effects — Linking: Menchero'); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.axvline(0,color='#5c6570',lw=.6); ax.legend(fontsize=8)
for ax in axes: ax.tick_params(axis='y',labelsize=8); ax.grid(axis='x',color='#e7ebef',lw=.6); ax.set_axisbelow(True)
fig.savefig(OUT/'sector_attribution.png',dpi=180,bbox_inches='tight'); plt.close(fig)
print(json.dumps(summary,indent=2))
