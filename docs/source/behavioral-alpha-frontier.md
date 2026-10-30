# Behavioral Alpha Frontier (BAF)

`brinsons.baf` is an **independent research implementation** inspired by the
[Behavioral Alpha Benchmark paper](https://www.northerntrust.com/content/dam/northerntrust/pws/nt/documents/asset-servicing/the-behavioral-alpha-benchmark.pdf)
and the [Growth Factor discussion](https://www.needhamfunds.com/wp-content/uploads/2026/03/46.-The-Growth-Factor-Vol.-46-Behavioral-Alpha-and-Stock-Market-Maestros.pdf).
It is an **independent reconstruction**. The papers do not disclose every
decision formula, the exact payoff scaling, or the official score mapping;
the formulas below define this project's calculations. It does not reproduce
Essentia's Decision Alpha, Behavioral Alpha Score, or peer ranks.

This module does not change the Brinson, linking, or multi-account APIs.
The BAF decision returns describe episodes, while Brinson effects reconcile
portfolio and benchmark returns. **Do not sum BAF decisions to explain a
portfolio's active return.**

## Input and episode construction

Provide a complete close-of-day grid with one row per portfolio, security,
and trading date. Required roles are portfolio, date, security, post-trade
shares, adjusted security price, and portfolio benchmark level. All names
are configurable through `prepare_baf(column_...=...)`; the input is never
renamed. A zero-share row is required on the liquidation date. Keep prices
for at least `forward_days` after exit to evaluate exit timing. Shares and
prices must use the same split-adjustment basis. The current model assumes
close-of-day executions and omits dividends, costs, cash, FX, and corporate
actions beyond adjusted prices. A return index that includes dividends can
be supplied as `price` if share units are made consistent with it.

For security $s$ and portfolio $p$, an episode is a maximal interval of
strictly positive observed shares. It begins when $q_{p,s,t}>0$ and the
previous observed share count is zero; it ends at the first later observed
zero. A subsequent purchase starts a new episode. Open episodes are excluded
by default or rejected with `open_episode_policy="error"`.

`calculate_baf(window_start=..., window_end=...)` includes only episodes
whose entry **and** exit fall within the optional window, while retaining
later price rows for exit evaluation. This is a simple completed-episode
window rule. The paper's 36-month quarterly rolling analysis and data
coverage criteria are research design details, not implemented eligibility
rules here. Truncating the raw grid at a window boundary can create false
episodes; prepare from full history and then select the window.

## Seven decisions and equations

Let $P_t$ be adjusted security price, $B_t$ the nominated benchmark level,
$Q_t=P_t/B_t$ the relative price, $q_t$ observed post-trade shares, $e$ entry,
and $x$ first zero-share exit date. Decision values $y$ are decimal return
differences or dimensionless decision returns. A positive value means the
observed choice beat the stated comparison. Phase definitions and
denominators are specified by this project.

| Decision function | Per-episode calculation |
|---|---|
| `stock_picking_decisions` | $y_{\rm pick}=(P_x/P_e-1)-(B_x/B_e-1)$ |
| `sizing_decisions` | $y_{\rm size}=R_{\rm MWR}(q)-R_{\rm MWR}(q^{\rm eq})$ |
| `entry_timing_decisions` | $y_{\rm entry}=\overline{Q}_{e+1:e+N}/Q_e-1$ |
| `scaling_in_decisions` | $y_{\rm in}=(q_bQ_e-\sum_{t=e}^{b}\Delta q_t^+Q_t)/(q_bQ_e)$ |
| `size_adjusting_decisions` | $y_{\rm adjust}=\sum_{t=b}^{u-2}(q_t-q_b)(Q_{t+1}-Q_t)/(q_bQ_b)$ |
| `scaling_out_decisions` | $y_{\rm out}=\sum_{t=u}^{x-1}q_t(Q_{t+1}-Q_t)/(q_{u-1}Q_u)$ |
| `exit_timing_decisions` | $y_{\rm exit}=1-\overline{Q}_{x+1:x+N}/Q_x$ |

The stock-picking formula uses endpoint adjusted prices, equivalent to a
buy-and-hold total-return index over an uninterrupted holding period. If
adjusted prices do not capture all distributions, it is not a full
time-weighted total return. The paper publishes the *concept* of episode
TWR relative to the benchmark, not its complete data-adjustment recipe.

For sizing, at each date $t$ let $V_t=\sum_{j:q_{j,t}>0}q_{j,t}P_{j,t}$ and
$n_t$ be the number of active securities. The package's explicit
equal-active-weight counterfactual is

$$
q^{\rm eq}_{s,t}=\begin{cases}V_t/(n_tP_{s,t})&q_{s,t}>0,\\0&\text{otherwise.}\end{cases}
$$

For either share path $a_t$, the close-of-day cash flow is
$C_t=-(a_t-a_{t-1})P_t$, with $a_{e-1}=a_x=0$. The episode money-weighted
return $R_{\rm MWR}$ solves

$$
\sum_{t=e}^{x}\frac{C_t}{(1+R_{\rm MWR})^{(d_t-d_e)/(d_x-d_e)}}=0,
$$

where $d_t$ is the calendar date in days. The code bisects on
$\log(1+R_{\rm MWR})$ and omits an episode if no root is bracketed.
The original paper does not specify its exact cash-flow timing, benchmark
rebalancing, or root-selection convention, so this is a documented
interpretation of its published sizing baseline.

For phase decisions, $b$ is the first active date on which shares reach
`phase_fraction` (default 0.9) of the episode peak. $u$ is the first later
date below that threshold with no subsequent recovery above it. The build
decision is omitted if it has no build or includes a sale. Size adjusting
requires a middle-phase share change; scaling out requires a partial
reduction. `forward_days=21` counts **observed trading rows**, not calendar
days. Incomplete forward windows are omitted. This phase model is intentionally
simple and may disagree with another reasonable decision taxonomy.

## Frontier coordinates

For one portfolio and one decision type, with decision values $y_i$:

$$
h=\frac{\#\{i:y_i\ge0\}}{n},\qquad
P=\frac{\operatorname{mean}(y_i\mid y_i\ge0)}
        {|\operatorname{mean}(y_i\mid y_i<0)|}.
$$

These are the paper's hit-rate and payoff definitions. Zero is counted as a
hit. Payoff is undefined when either side is absent; `payoff` and
`scaled_payoff` are then null. The package uses the explicitly **assumed**
log-logistic scale

$$
S_k(P)=\frac{1}{1+e^{-k\ln P}}=\frac{P^k}{1+P^k},\qquad
P=\left(\frac{S}{1-S}\right)^{1/k},\qquad k>0.
$$

The continuous value at $P=0$ is $S_k(0)=0$. The default $k=1.5$ is an
**illustrative shape parameter**, estimated informally from the published
figure; it is neither a published coefficient nor a fitted reconstruction.
The exact Essentia scaling transform is undisclosed. $S_k(1)=0.5$ for every
$k$. `decision_metrics` and `calculate_baf` expose
`payoff_scale_exponent`; `plot_baf` defaults to the exponent stored in its
`BAFResult`, and an explicit override rescales its plotted points and
background together.

### Data-independent color bands and indifference curves

Let $L=|\operatorname{mean}(y\mid y<0)|$. When both wins and losses occur,
the sample mean decision value satisfies

$$
\operatorname{mean}(y)
=L\underbrace{\bigl[hP-(1-h)\bigr]}_{E(h,P)}.
$$

The chart colors the **normalized edge** $E(h,P)=hP-(1-h)$ over a fixed
$(h,S)$ grid, independently of the observed managers. This is a
dimensionless normalization by each group's mean loss. Equal $E$ does not
imply equal raw mean return when their $L$ values differ. At a fixed level
$E=c$, the indifference curve is

$$
P_c(h)=\frac{c+1-h}{h},\qquad
S_c(h)=S_k\!\left(\frac{c+1-h}{h}\right),
$$

where $h>0$ and $P_c(h)\ge0$. The emphasized chance boundary has $c=0$:

$$
S_0(h)=\frac{1}{1+\left(h/(1-h)\right)^k}.
$$

It passes through $(0.5,0.5)$ and is curved for $k\ne1$; for example,
$S_0(0.25)\approx0.839$ when $k=1.5$. Negative bands are lower/left and
positive bands higher/right. `plot_baf` uses `contourf` for the color bands,
draws white level boundaries, and overlays the thick dashed $E=0$ curve.
The defaults `contour_levels=(-1,-0.75,-0.5,-0.25,0,0.25,0.5,1,2,5,50)`,
`grid_size=161`, and `cmap="Purples"` can be changed at the function entrance.
The source paper calls its bands illustrative; neither its exact scale nor
chosen levels are disclosed. These contours are **our reproducible model**,
not Essentia's proprietary EBAF curves.

`all_decisions` pools the seven decision records for descriptive
viewing. Different decision families can have different units, denominators,
and overlap; the pooled point is **not** an Essentia score or an additive
measure. `minimum_decisions` can hide small samples. No official 0–100 score
or peer rank is inferred from the publication.

### Peer-change summary table

The paper's Exhibit 8 lays out three changes in the **percentage of
portfolios**: adding value, hit rate above 50%, and payoff above 1. The
separate `baf_change_table(previous, current)` function follows that layout
but calculates these quantities from this package's metrics. For each
decision family $d$ and a paired set of eligible portfolios $M_d$, let

$$
A_{w,d}=\frac{1}{|M_d|}\sum_{p\in M_d}{\bf1}[\overline y_{p,w,d}>0],\qquad
H_{w,d}=\frac{1}{|M_d|}\sum_{p\in M_d}{\bf1}[h_{p,w,d}>0.5],\qquad
Q_{w,d}=\frac{1}{|M_d|}\sum_{p\in M_d}{\bf1}[P_{p,w,d}>1].
$$

The displayed cells are $100(X_{\rm current,d}-X_{\rm previous,d})\%$
for $X\in\{A,H,Q\}$: the current percentage minus the previous percentage,
not a relative growth rate. Only portfolio/decision pairs
present in both windows with defined payoff in both enter $M_d$. The
returned Polars table includes `n_common_portfolios` and the three `*_change_percent`
columns. `chart_baf_change_table` renders a green/pink matrix; green means a
larger share, pink a smaller share. It has configurable source column names,
title, and figure size. The source paper's values, quarterly eligibility
filter, and proprietary Decision Alpha are **not** reproduced.

The [synthetic peer-change table](../../outputs/baf/synthetic_peer_changes.png)
uses two nonoverlapping completed-episode windows with three fictional
managers. Changes therefore occur in coarse $33.3\%$ increments and
must not be interpreted as estimates of real manager behavior.

## Python workflow

```python
import polars as pl
from datetime import date
from brinsons import (prepare_baf, calculate_baf, plot_baf,
                      baf_change_table, chart_baf_change_table)

grid = pl.read_csv("data/baf/synthetic_daily_positions.csv", try_parse_dates=True)
prepared = prepare_baf(grid)
result = calculate_baf(prepared, entry_forward_days=21, exit_forward_days=21,
                       payoff_scale_exponent=1.5)
print(result.prepared.episodes)
print(result.decisions)
print(result.metrics)
fig, axes = plot_baf(result, contour_levels=(-1, -0.5, 0, 0.5, 1, 2, 5, 50))

early = calculate_baf(prepared, window_end=date(2025, 4, 1))
late = calculate_baf(prepared, window_start=date(2025, 4, 1))
changes = baf_change_table(early, late)
table_fig, table_ax = chart_baf_change_table(changes)
```

The generated [synthetic frontier image](../../outputs/baf/synthetic_frontier.png)
shows the fixed color bands, curved chance boundary, and manager points.

Install notebook dependencies with `python -m pip install -e ".[tutorial]"`.
Run `python scripts/generate_baf_data.py` to regenerate the deterministic
fictional grid. The [English notebook](../../tutorials/en/behavioral_alpha_frontier.ipynb)
and [中文教程](../../tutorials/zh/behavioral_alpha_frontier.ipynb) show the
separate decision functions and frontier. Install Matplotlib for the chart:
`python -m pip install -e ".[plot]"`.

## Interpretation limits

The synthetic managers, securities, prices, and decisions are fictional.
The paper's empirical percentages and published 50/56 score examples are
not expected results for these data. A plotted point reflects a choice of
episode construction, adjusted prices, counterfactuals, forward window,
sample, and missing-data policy. The decision rules, payoff scale,
and combined point are specified transparently in this project.
