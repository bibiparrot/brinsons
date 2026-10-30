# Data and column contract

For each period $t$ and segment $i$, the input supplies portfolio and benchmark weights $w_{P,i,t}$ and $w_{B,i,t}$, and segment returns $r_{P,i,t}$ and $r_{B,i,t}$. Each `(period, segment)` pair is unique; each side's period weights sum to one. The calculation retains input column names and stores their roles in `result.columns` (`ColumnNames`). It does not rename the input frame.

| Role | Default column | Meaning |
|---|---|---|
| Time key | `period` | Chronologically sortable period |
| Segment key | `segment` | Unique category within period |
| Weights | `portfolio_weight`, `benchmark_weight` | Decimal weights |
| Returns | `portfolio_return`, `benchmark_return` | Simple decimal segment returns |

Every core calculation exposes the corresponding `column_...` keyword arguments. For example:

```python
from brinsons import fachler

result = fachler(
    sector_data,
    column_period="Period",
    column_segment="Category",
    column_portfolio_weight="PortfolioWeight",
    column_benchmark_weight="BenchmarkWeight",
    column_portfolio_return="PortfolioReturn",
    column_benchmark_return="BenchmarkReturn",
)
assert result.columns.segment == "Category"
```

Alternatively, supply `columns=ColumnNames(...)` to choose input **and output effect** field names. Do not combine a nondefault `columns` object with individual column arguments. The generated summary has its own `summary_*` names in `ColumnNames`. `portfolio_contribution` and `benchmark_contribution` are $w_P r_P$ and $w_B r_B$ at detail level. The summary has period portfolio return, benchmark return, excess return, and the effect totals.

`prepare(portfolio, benchmark)` aligns separate side frames. For a segment absent on one side, it uses zero weight and the present side's return; this makes selection zero for that row. That is a reporting convention, so review it for off-benchmark positions. Effects require a chosen policy for missing conditional segment returns; exact mathematical closure does not establish that policy.

The [API reference](api.rst) lists all keyword arguments. Covariance-driven methods use complete long-form frames with `row`, `column`, `covariance` by default; all entries, including zeros, must be present.
