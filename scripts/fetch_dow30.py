import json, math, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

tickers = ['AAPL','AMGN','AMZN','AXP','BA','CAT','CRM','CSCO','CVX','DIS','GS','HD','HON','IBM','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','NVDA','PG','SHW','TRV','UNH','V','WMT','GOOGL']
out = Path(__file__).resolve().parents[1] / 'data'
out.mkdir(parents=True, exist_ok=True)
print('Downloading daily prices', flush=True)
prices = yf.download(tickers+['^DJI'], period='1y', interval='1d', auto_adjust=True, group_by='ticker', threads=True, progress=False, timeout=20)
rows=[]
for symbol in tickers+['^DJI']:
    if symbol not in prices.columns.get_level_values(0):
        print('Missing prices:',symbol,flush=True)
        continue
    frame=prices[symbol].dropna(how='all')
    for date,row in frame.iterrows():
        def n(name):
            value = row.get(name)
            return None if value is None or pd.isna(value) else float(value)
        rows.append([date.strftime('%Y-%m-%d'),symbol,n('Open'),n('High'),n('Low'),n('Close'),n('Volume')])
    print(symbol, len(frame), flush=True)
metadata=[]
for symbol in tickers:
    try:
        info=yf.Ticker(symbol).info
        metadata.append([symbol,info.get('longName') or info.get('shortName'),info.get('sector'),info.get('industry'),info.get('sectorKey'),info.get('industryKey')])
        print(symbol,info.get('sector'),info.get('industry'),flush=True)
    except Exception as e:
        metadata.append([symbol,None,None,None,None,None])
        print('METADATA ERROR',symbol,str(e),flush=True)
    time.sleep(.25)
result={'source':'Yahoo Finance via yfinance '+yf.__version__, 'retrieved_at_utc':datetime.now(timezone.utc).isoformat(), 'period':'1y', 'adjusted':True,'tickers':tickers,'prices':rows,'metadata':metadata}
(out/'dow30_yfinance_data.json').write_text(json.dumps(result,allow_nan=False),encoding='utf8')
print('Done. Price rows:',len(rows),'Metadata:',len(metadata),flush=True)
