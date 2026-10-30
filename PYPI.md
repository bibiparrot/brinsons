# brinsons

**Polars-based portfolio performance attribution** for single-period and
multiple-period Brinson analysis, multi-account attribution, currency and
factor methods, and an independently specified Behavioral Alpha Frontier.

Python 3.11+ · Polars is the only core runtime dependency · Matplotlib is
optional for charts.

![Behavioral Alpha Frontier](https://raw.githubusercontent.com/bibiparrot/brinsons/main/assets/synthetic_frontier.png)

![Brinson excess-return waterfall](https://raw.githubusercontent.com/bibiparrot/brinsons/main/assets/single_period_waterfall.png)

## Install

```sh
python -m pip install brinsons
# With plotting tools:
python -m pip install "brinsons[plot]"
```

## Start with Brinson attribution

```python
import polars as pl
from brinsons import calculate_single_period, calculate_multiple_period
from brinsons import bhb, fachler, link_carino

facts = pl.DataFrame({
    "period": [1, 1, 2, 2],
    "segment": ["Equity", "Bonds", "Equity", "Bonds"],
    "portfolio_weight": [.6, .4, .55, .45],
    "benchmark_weight": [.5, .5, .5, .5],
    "portfolio_return": [.12, .04, .03, .01],
    "benchmark_return": [.10, .06, .02, .015],
})

single = calculate_single_period(facts, period=1, attribution=bhb)
multiple = calculate_multiple_period(facts, attribution=fachler, linking=link_carino)
print(single.summary)
print(multiple.linked.summary)
```

Input column names are configurable through `column_...` parameters; source
columns do not need to be renamed. Returns and effects use decimal units:
`0.01` is 1% or 100 basis points.

## Explore

- [English README](https://github.com/bibiparrot/brinsons/blob/main/README.md) · [中文 README](https://github.com/bibiparrot/brinsons/blob/main/README.zh-CN.md)
- [Bilingual Jupyter tutorials](https://github.com/bibiparrot/brinsons/tree/main/tutorials)
- [Sphinx method guide and equations](https://github.com/bibiparrot/brinsons/tree/main/docs/source)
- [Python API](https://github.com/bibiparrot/brinsons/blob/main/docs/source/api.rst)

The Behavioral Alpha Frontier calculations are independently defined. The
source papers do not disclose every proprietary decision algorithm or the
official score mapping; this package does not claim to reproduce Essentia's
scores.
