# brinsons

**English** · [简体中文](README.zh-CN.md)

Portfolio performance attribution with Polars expressions: single-period Brinson,
five multiple-period linking methods, multi-account decision attribution,
currency and factor extensions, and an independently specified Behavioral Alpha
Frontier (BAF). **Polars is the only core runtime dependency**; Matplotlib is
optional for charts. The [Sphinx method guide](docs/source/index.md) gives the
equations, API, input contracts, and verification notes.

## See it in action

The images below come from the **executed Jupyter tutorials**. The BAF managers
and multi-account portfolios are synthetic; the Dow 30 tutorial uses the local
Yahoo Finance research snapshot. Open the [tutorial index](tutorials/README.md)
for paired English and Chinese notebooks.

### Behavioral Alpha Frontier

Seven decision families, a curved chance boundary, and normalized-edge color
bands. The accompanying table compares two windows using the current
percentage minus the previous percentage for the same eligible managers.
The methods and color scale are fully specified in the [BAF guide](docs/source/behavioral-alpha-frontier.md).

![BAF frontier with indifference bands](assets/synthetic_frontier.png)

![BAF peer-change table](assets/synthetic_peer_changes.png)

[Explore the BAF tutorial](tutorials/en/behavioral_alpha_frontier.ipynb)

### Brinson attribution and linking

Compare BHB and Brinson–Fachler under Cariño, GRAP, Frongello, and Menchero
linking, then trace allocation, selection, and interaction into total active
return. Geometric linking has a separate comparison in the notebook.

![Multiple-period method comparison](assets/multiple_period_comparison.png)

![Single-period excess-return waterfall](assets/single_period_waterfall.png)

[Single-period tutorial](tutorials/en/single_period_attribution.ipynb) ·
[Multiple-period tutorial](tutorials/en/multiple_period_attribution.ipynb)

### Dow 30 and multi-account workflows

Inspect category returns, weights, and active effects from the Dow 30 research
snapshot, or follow account allocation into lower-level decisions.

![Dow 30 returns, weights, and active effects](assets/dow30_interactive.png)

![Multi-account attribution comparison](assets/multi_account_comparison.png)

[Dow 30 tutorial](tutorials/en/dow30_brinson_interactive.ipynb) ·
[Multi-account tutorial](tutorials/en/multi_account_attribution.ipynb)

Returns, weights, and effects use decimal units: `0.01` means 1% or 100 basis
points. Inputs must already be aligned to meaningful periods and segment
returns. The [extended mathematical note](docs/papers/brinson-attribution-methods.md)
records additional conventions.

## Install and run

Install the published package with `python -m pip install brinsons` or add
`[plot]` for chart functions. To work from this source checkout:

```sh
python -m pip install -e .
python examples/generate.py
```

Build the Sphinx HTML guide with `python -m pip install -e ".[docs]"`
followed by `python -m sphinx -b html -W --keep-going docs/source docs/_build/html`.

The example generates deterministic source data and CSV output in `data/examples/`. The tutorials read reusable Dow 30 CSV tables from `data/dow30_brinson/`; the Yahoo snapshot is `data/dow30_yfinance_data.json`, and the price/sector workbook is `data/dow30/Dow30_prices_sectors.xlsx`. Generated analysis reports, charts, and the analysis workbook remain in `outputs/dow30_brinson/`.

```python
import polars as pl
from brinsons import calculate_single_period, calculate_multiple_period, fachler, bhb, link_carino

facts = pl.DataFrame({
    "period": [1, 1, 2, 2],
    "segment": ["Equity", "Bonds", "Equity", "Bonds"],
    "portfolio_weight": [.6, .4, .55, .45],
    "benchmark_weight": [.5, .5, .5, .5],
    "portfolio_return": [.12, .04, .03, .01],
    "benchmark_return": [.10, .06, .02, .015],
})
single = calculate_single_period(facts, period=1, attribution=bhb)
print(single.detail)                  # period 1 segment effects
print(single.summary)                 # period 1 returns and effects

multiple = calculate_multiple_period(facts, attribution=fachler, linking=link_carino)
print(multiple.periods.summary)       # returns and effects for each period
print(multiple.linked.detail)         # linked segment/period effects
print(multiple.linked.summary)        # full-horizon returns and effects
```

`calculate_single_period` requires exactly one period or an explicit `period` selection. `calculate_multiple_period` requires at least two periods and returns a `MultiplePeriodAttribution` with two `AttributionResult` fields, `periods` and `linked`. Choose the `attribution` callable (`bhb` or `fachler`) and, for multiple periods, the `linking` callable (`link_carino`, `link_grap`, `link_frongello`, `link_menchero`, or `link_geometric`). The defaults are `fachler` and `link_menchero`. `effects=2` or `effects=3` selects the presentation; both functions also accept the same `column_...` keyword arguments as `bhb` and `fachler`. Compatible custom callables may be supplied. The existing `bhb`, `fachler`, and `link_...` functions remain available separately.

Input columns are `period`, `segment`, `portfolio_weight`, `benchmark_weight`, `portfolio_return`, `benchmark_return`. Each period/segment key must be unique and each side's weights sum to one. The period key must sort chronologically, for example an integer or date; the function does not infer or validate period lengths. `prepare(portfolio, benchmark)` aligns separate frames with columns `period`, `segment`, `weight`, `return`. An absent segment receives zero weight and the present side's return, so there is no artificial selection on that row. This convention changes effect assignment for off-benchmark/absent segments; select it only if appropriate for your reporting policy.

### Custom input columns

Public functions accept keyword-only `column_...` parameters, with the names above as defaults. For example, sector data can be used without renaming the input frame:

```python
result = fachler(
    sector_data,
    column_period="Period",
    column_segment="Category",
    column_portfolio_weight="PortfolioWeight",
    column_benchmark_weight="BenchmarkWeight",
    column_portfolio_return="PortfolioReturn",
    column_benchmark_return="BenchmarkReturn",
)
linked = link_carino(result)
```

`result.detail` keeps the supplied input names, and `result.summary` keeps the supplied period name. `result.columns` is a `ColumnNames` instance that records the names used by detail and summary. You can also pass `columns=ColumnNames(period="Period", segment="Category", ...)` instead of individual `column_...` arguments. The effect and summary field names can be set on that class; newly calculated fields use its defaults otherwise. The class is carried through attribution, linking, and reporting. One input can supply more than one role: for example, set both `column_portfolio_return="Return"` and `column_benchmark_return="Return"` when both refer to the same asset return. `prepare` has side-specific overrides such as `column_portfolio_weight` and `column_benchmark_weight` when its two frames use different schemas; it creates distinct output weight and return fields. Hierarchy inputs have `column_child` and `column_parent`. Currency functions expose their local, currency, and effect column names; `currency_premiums` additionally accepts `market_columns` and `currency_columns` dictionaries for independently named market and currency inputs. Hentschel methods use asset and covariance names supplied in their arguments. Calculated output fields that collide with input fields raise `AttributionError`.

The separate linking functions are `link_carino`, `link_grap`, `link_frongello`, `link_menchero`, and `link_geometric`. Arithmetic results include linked contributions per segment and period. Their summary reconciles to the difference of compounded returns. GRAP and Frongello here give the same factors; Menchero correctly handles equal cumulative returns with unequal period active returns. Geometric returns a period-level detail and a full-horizon summary with multiplicative reconciliation; it combines selection and interaction. Linking requires at least two chronological periods, valid detail matching the summary, and each portfolio and benchmark period return greater than −100%. Each result describes one portfolio and benchmark stream; group separate streams before calling. Two-effect inputs and three-effect inputs should not be mixed across periods.

### Markdown reports and charts

Each report and chart has a separate single-period and multiple-period function:

| Report | Single period | Multiple periods |
| --- | --- | --- |
| Category weights | `print_single_period_category_weights` | `print_multiple_period_category_weights` |
| Category returns | `print_single_period_category_returns` | `print_multiple_period_category_returns` |
| Category attribution | `print_single_period_category_attribution` | `print_multiple_period_category_attribution` |
| Total attribution | `print_single_period_total_attribution` | `print_multiple_period_total_attribution` |
| Attribution summary | `print_single_period_attribution_summary` | `print_multiple_period_attribution_summary` |
| Returns chart | `chart_single_period_category_returns` | `chart_multiple_period_category_returns` |
| Weights chart | `chart_single_period_category_weights` | `chart_multiple_period_category_weights` |
| Attribution chart | `chart_single_period_category_attribution` | `chart_multiple_period_category_attribution` |
| Excess-return waterfall | `chart_single_period_excess_return_waterfall` | `chart_multiple_period_excess_return_waterfall` |

The Markdown functions display formatted explanations and tables, and return the Markdown text. In Jupyter they render Markdown; in a terminal they print its source. The chart functions return Matplotlib `(figure, axes)` objects. Install plotting support with `python -m pip install -e ".[plot]"`.

```python
from brinsons import print_single_period_category_weights, print_multiple_period_category_weights
from brinsons import chart_single_period_category_returns, chart_multiple_period_category_returns
print_single_period_category_weights(result, period=1)
print_multiple_period_category_weights(result)
fig, ax = chart_single_period_category_returns(result, period=1)
fig, ax = chart_multiple_period_category_returns(result)
```

All functions accept an `AttributionResult` from `bhb` or `fachler` and read its `ColumnNames` metadata. Single-period functions require a `period`; multiple-period functions require at least two periods. Multi-period attribution reports and charts accept `linking=link_carino`, `link_grap`, `link_frongello`, or `link_menchero`; the default is `link_menchero`. Geometric linking is supported by the total-attribution and summary reports, and by the aggregate waterfall, all of which show its multiplicative cross term. Category-attribution charts require segment-level arithmetic effects, so they reject geometric linking. Every multiple-period report and chart title identifies the selected method. Category returns compound each period's category return, with within-category weights reset each period; weights and category returns say `Linking: none` because they do not use an attribution linker. Average weights treat an absent category as zero; compounded category-return reports require every category to be present in every period. For an externally constructed result with custom column names, pass the corresponding `column_...` keyword arguments. The original functions with an optional `period` remain available for existing callers. See [the bilingual tutorial index](tutorials/README.md), with separate [single-period](tutorials/en/single_period_attribution.ipynb) and [multiple-period](tutorials/en/multiple_period_attribution.ipynb) notebooks for examples.

```python
from brinsons import bhb, link_carino, calculate_multiple_period
from brinsons import print_multiple_period_total_attribution, chart_multiple_period_excess_return_waterfall

run = calculate_multiple_period(data, attribution=bhb, effects=3, linking=link_carino)
print_multiple_period_total_attribution(run.periods, linking=link_carino)
fig, ax = chart_multiple_period_excess_return_waterfall(run.periods, linking=link_carino)
```

The waterfall charts show effects followed by the linked active return, in basis points. Arithmetic linking can display each category within allocation, selection, and interaction; `by_category=False` shows aggregate effects. Geometric linking automatically uses the aggregate view with allocation, selection, their cross term, and relative active return; explicitly requesting `by_category=True` raises an error. The arithmetic total is the difference of compounded portfolio and benchmark returns, while the geometric total is their compounded wealth ratio minus one.

`roll_up(detail, hierarchy, effects=None)` sums leaf effects through arbitrary depth. `hierarchy` has unique `child` and nullable `parent` columns; root nodes need not appear as children. The output includes leaf and ancestor rows, which must not be summed together. Pass `effects` explicitly for linked effects or the additive currency columns.

`currency_attribution(facts)` uses the simplified **additive** segment currency split of pybrinson. It requires `local_return` and `currency_return` alongside the basic columns, with `portfolio_return == local_return + currency_return` and a shared segment currency contribution for portfolio and benchmark. It does not account independently for hedges or forward rates. For simple returns, ordinary FX multiplication contains a cross term; inputs must use a consistent additive contribution convention.

`currency_premiums(markets, currencies)` is a separate perfattr-inspired log-return model. Each frame has the standard weights, returns, period and segment; markets also supply `local_cash_return`. Market and currency weights are independent and each side must sum to one in each grid. The function returns `(market_detail, currency_detail, period_summary)`; the effects and modeled return are log quantities, not an independently measured simple accounting return. It uses aligned data, one common local-cash return per market, and already supplied net currency exposures; it does not price hedges. `roll_up_currency_premiums(market_detail, currency_detail, period_summary)` returns market and currency identifier totals and chronologically cumulative log effects.

## Multi-account attribution

The independent `src/brinsons/multi_account.py` module provides three calculation entrances: `macro_micro_attribution` (account allocation followed by within-account Brinson), `value_weighted_total_attribution` (one pooled portfolio and weighted benchmark), and `hierarchical_decision_attribution` (successive account, sector, security-weight, and security-return notionals). It accepts custom input column names without renaming the input frame. The synthetic [multi-account data](data/multi_account/positions.csv) can be regenerated with `python scripts/generate_multi_account_data.py`; the [English](tutorials/en/multi_account_attribution.ipynb) and [Chinese](tutorials/zh/multi_account_attribution.ipynb) interactive tutorials compares all three methods on that same data. See the [multi-account equations](docs/source/multi-account.md) for the exact integration and reconciliation identities.

For multiple periods, `link_macro_micro_attribution` and `link_hierarchical_decision_attribution` take an explicit arithmetic `linking=` function (Cariño, GRAP, Frongello, or Menchero); `value_weighted_total_attribution` returns a standard `AttributionResult` that can be passed directly to a linker. The additive account/decision tables reject geometric linking.

## Verification

```sh
python -m pip install -e ".[test]"
python -m pytest -q
```

Tests generate seeded portfolio/benchmark data and compare against the MIT-licensed [pybrinson test reference](tests/vendor/README.md) included under `tests/vendor`. They compare BHB/BF segment and period effects, all five aggregate linking methods, and additive currency attribution. They also test Menchero's edge case, hierarchy, invalid inputs, the independently modeled currency example, multi-account methods, and BAF equations and charts. The reference is **test-only**, not a runtime dependency.

## Scope and limitations

This is not an API-compatible reimplementation of all `perfattr` functionality. It does not yet implement calendar/holiday-aware period consolidation, authoritative contribution and fee handling, effective-dated mappings, performance file readers, classification metadata, or the full `perfattr` result schemas. It also does not implement a fully general Karnosky–Singer hedge or forward model. Inputs with external cash flows, changing weights within a period, zero-weight accounting contributions, or missing segment returns require upstream accounting and preparation.

## Hentschel regression-based attribution

The project includes separate functions for restricted factor attribution, Brinson interaction reallocation, covariance propagation, precision-weighted period linking, and entry-specific reconciliation. See [Hentschel methods](docs/source/hentschel.md) for equations, covariance schemas, examples, and a reconciliation issue in the paper's single-constraint Brinson formulation.
