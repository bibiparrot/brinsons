# Quick start

Install the core package and, separately, the documentation tools:

```console
python -m pip install -e .
python -m pip install -e ".[docs]"
python -m sphinx -b html -W --keep-going docs/source docs/_build/html
```

Open `docs/_build/html/index.html`. The documentation build does not download market data. For reports and charts, install `.[plot]`.

## Single period

```python
import polars as pl
from brinsons import calculate_single_period, fachler

facts = pl.DataFrame({
    "period": [1, 1, 2, 2],
    "segment": ["Equity", "Bonds", "Equity", "Bonds"],
    "portfolio_weight": [.6, .4, .55, .45],
    "benchmark_weight": [.5, .5, .5, .5],
    "portfolio_return": [.12, .04, .03, .01],
    "benchmark_return": [.10, .06, .02, .015],
})
single = calculate_single_period(facts, period=1, attribution=fachler, effects=3)
print(single.detail)
print(single.summary)
```

`calculate_single_period` requires one period; pass `period` when the frame has several. `bhb` and `fachler` are separate calculation functions. Set `effects=2` to absorb interaction into selection.

## Multiple periods

```python
from brinsons import calculate_multiple_period, link_carino

multiple = calculate_multiple_period(
    facts, attribution=fachler, linking=link_carino, effects=3
)
print(multiple.periods.summary)  # one row per period
print(multiple.linked.summary)   # full horizon
```

`calculate_multiple_period` requires at least two periods. Its `linking` callable is explicit; the default is `link_menchero`. Choose one of `link_carino`, `link_grap`, `link_frongello`, `link_menchero` or `link_geometric`. The first four reconcile **arithmetic** compounded excess. Geometric linking reports **relative** compounded excess and obeys a multiplicative identity.

```{note}
The `period` column must sort chronologically. A result represents one portfolio and one benchmark stream. Returns must use the same period boundaries as weights; the library does not infer flows, fees, or period lengths.
```
