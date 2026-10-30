# Reports and charts

The reporting helpers take an `AttributionResult` from `bhb` or `fachler`. Markdown functions display a formatted table in Jupyter and return its Markdown source. Chart functions return Matplotlib `(figure, axes)`; install `brinsons[plot]` to use them. Input names come from `result.columns` or explicit `column_...` arguments.

| View | Single-period function | Multiple-period function |
|---|---|---|
| Category weights | `print_single_period_category_weights` | `print_multiple_period_category_weights` |
| Category returns | `print_single_period_category_returns` | `print_multiple_period_category_returns` |
| Category attribution | `print_single_period_category_attribution` | `print_multiple_period_category_attribution` |
| Total attribution | `print_single_period_total_attribution` | `print_multiple_period_total_attribution` |
| Attribution summary | `print_single_period_attribution_summary` | `print_multiple_period_attribution_summary` |
| Returns bars | `chart_single_period_category_returns` | `chart_multiple_period_category_returns` |
| Weights bars | `chart_single_period_category_weights` | `chart_multiple_period_category_weights` |
| Attribution bars | `chart_single_period_category_attribution` | `chart_multiple_period_category_attribution` |
| Excess-return waterfall | `chart_single_period_excess_return_waterfall` | `chart_multiple_period_excess_return_waterfall` |

Single-period functions require `period=...`. Multi-period attribution views and the waterfall expose `linking=...`, defaulting to `link_menchero`. Their titles name the selected linking method. The weights and returns views use no attribution linker and label that fact. Category attribution charts need linked **segment** effects, so they accept arithmetic linkers only. The aggregate geometric total, summary and waterfall use $G=A_G+S_G+A_GS_G$; the cross term is shown separately.

```python
from brinsons import (
    fachler, link_carino,
    print_multiple_period_attribution_summary,
    chart_multiple_period_excess_return_waterfall,
)

periods = fachler(facts)
markdown = print_multiple_period_attribution_summary(periods, linking=link_carino)
fig, ax = chart_multiple_period_excess_return_waterfall(
    periods, linking=link_carino, by_category=False
)
```

The [bilingual tutorial index](tutorials.md) separates single-period attribution from multiple-period linking. The Dow 30 interactive tutorial shows the Yahoo-data workflow. Their reusable CSV inputs live in `data/dow30_brinson/`.

For arithmetic waterfalls, the total shown is $E=A+S+I$. For geometric, it is the relative compounded excess $G$. The chart's unit conversion uses basis points: $\mathrm{bp}=10^4\times\text{decimal return}$.
