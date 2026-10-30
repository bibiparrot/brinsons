# Dow 30 Brinson attribution example

This is a Python adaptation of the [MathWorks Brinson workflow](https://www.mathworks.com/help/releases/R2023a/finance/workflow-for-performance-attribution-using-brinson-model.html). It uses Yahoo Finance prices, the local `brinsons` library, and an illustrative portfolio. The workbook contains the six-column asset table, sector and period results, and the monthly prices used to calculate returns.

The 12 complete monthly periods run from the September 30, 2025 closing price to the September 30, 2026 closing price. The latest partial month is excluded. Each monthly return is the ratio of the stock's adjusted month-end close to its previous adjusted month-end close, less one. Yahoo sector classifications and the October 29, 2026 constituent snapshot are applied to all historical months. This is a fixed current universe, not a historically accurate membership or sector backtest.

The benchmark has a weight of 1/30 in each stock at the start of every month, as in the MathWorks example. **It is not the price-weighted Dow Jones Industrial Average.** The illustrative portfolio weights use fixed sector tilts and deterministic ticker/quarter factors, without using the returns being analyzed. They are target weights reset at each month start, with the factor schedule changing every three periods. They do not represent an actual portfolio, transactions, fees, or a recommendation.

Within each period, portfolio and benchmark sector returns are weighted averages of the underlying stock returns. `brinsons.fachler` computes allocation, selection and interaction; `brinsons.link_menchero` links their period/sector results to the difference between the compounded portfolio and benchmark returns. Average sector weights are simple averages across months. Each compounded sector-return series represents a monthly reset of within-sector weights, not a separate continuously held sector portfolio.

| Full-period result | Return or effect |
|---|---:|
| Synthetic portfolio | 14.3557% |
| Equal-weight benchmark | 14.3573% |
| Active return | −0.0016 percentage points |
| Allocation | +0.6768 percentage points |
| Selection | −0.6223 percentage points |
| Interaction | −0.0560 percentage points |

Allocation and selection nearly offset. Consumer Cyclical was the largest positive linked sector contribution at approximately +0.4061 percentage points; Industrials was the largest negative at approximately −0.2422 percentage points. Figures can differ in the final decimal due to display rounding.

The script [dow30_brinson_analysis.py](../../scripts/dow30_brinson_analysis.py) downloads the adjusted price history with `yfinance`, reads the cached Yahoo sector metadata created by `scripts/fetch_dow30.py`, generates the asset table, calls `brinsons`, and saves the analysis CSVs and chart. Run it from the project root after installing `yfinance`, `matplotlib`, Polars and `brinsons`. Rerunning downloads current Yahoo data and can change the date range and results; refresh the cached metadata separately when a current constituent snapshot is needed.

Validation checked all 360 workbook asset rows against the generated table, recalculated each monthly return from the saved price observations, checked normalized weights, compared sector returns against direct asset-weighted totals, and verified the linked attribution identity. No MATLAB execution was performed, so the analysis follows its documented method rather than asserting numerical equivalence to a MATLAB run.
