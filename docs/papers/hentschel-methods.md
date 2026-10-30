# Hentschel: regression-based attribution

This note describes the implementation of Hentschel's *Brinson and Factors: A Unified Framework for Complete Portfolio Return Attribution* (July 12, 2026). See also [Brinson methods](brinson-attribution-methods.md). Each method uses an explicit covariance or precision metric: exact reconciliation does not validate that metric.

## Restricted generalized least squares

For estimates $\hat\theta$, covariance/metric $\Sigma$, and constraints $Q\theta=b$, the adjustment is

$$
\theta^*=\hat\theta+\Sigma Q^\mathsf T(Q\Sigma Q^\mathsf T)^{-1}(b-Q\hat\theta).
$$

Covariances are complete long-form Polars frames with columns `row`, `column`, `covariance`, including all zero entries. They must be finite, symmetric, positive semidefinite and keyed exactly to the relevant inputs. A singular system needing a nonzero adjustment raises `AttributionError`; no ridge or pseudoinverse is applied. Calculations use Polars expressions and a small standard-library linear solver; NumPy, pandas and SciPy are not required.

Input names are configurable with keyword parameters. For example, covariance columns `from`, `to`, and `value` can be supplied with `column_covariance_row="from"`, `column_covariance_column="to"`, and `column_covariance_value="value"`; asset and segment columns can be selected with `column_asset` and `column_segment`. Calculations use those names directly, and attribution results carry the selected period and segment names in `result.columns`. See the [README](../../README.md#custom-input-columns) for the general convention.

## Factor attribution

`factor_attribution_restricted(assets, factors)` implements equations (3), (6), and (7). For factor matrix $X$, diagonal regression weight matrix $W$, asset returns $r$ and active weights $w_A$,

$$
\begin{aligned}
\hat f&=(X^\mathsf TWX)^{-1}X^\mathsf TWr,&\Omega&=(X^\mathsf TWX)^{-1},\\
q&=X^\mathsf Tw_A,&b&=w_A^\mathsf Tr,\\
f^*&=\hat f+\Omega q(q^\mathsf T\Omega q)^{-1}(b-q^\mathsf T\hat f),&c_k^*&=q_k f_k^*.
\end{aligned}
$$

The asset frame contains unique `asset`, `return`, `portfolio_weight`, `benchmark_weight`, and a column for each factor. Optional positive `regression_weight` defaults to one. Each weight vector must sum to one. The returned frame includes unrestricted and adjusted factor returns and contributions; the restricted contributions sum to observed active return. The regression is cross-sectional and must be full rank. The paper's $\Omega$ is a precision metric; its interpretation as sampling covariance requires additional error-model assumptions.

```python
import polars as pl
from brinsons import factor_attribution_restricted

assets = pl.DataFrame({
    "asset": ["a", "b", "c"], "return": [.03, .01, .08],
    "portfolio_weight": [.7, .2, .1], "benchmark_weight": [.2, .3, .5],
    "intercept": [1., 1., 1.], "value": [-1., 0., 1.],
})
factor_effects = factor_attribution_restricted(assets, ["intercept", "value"])
```

## Brinson interaction reallocation

`brinson_interaction_restricted(facts, covariance)` implements the paper's single restriction, equations (14) and (20), for one period. With $m=[\mu_B;\mu_P]$, $d=w_P-w_B$, $q=[-d;d]$ and $I=q^\mathsf Tm$,

$$m^*=m-\Gamma q(q^\mathsf T\Gamma q)^{-1}I.$$

The covariance uses group-mean keys `B:<segment>` and `P:<segment>`. Output includes adjusted group means, allocation, selection, aggregate interaction, and `reconciliation_residual` (original active return minus adjusted allocation and selection).

**The single restriction does not generally preserve the original portfolio and benchmark returns.** In the paper's two-group example, with $w_B=(.5,.5)$, $w_P=(.6,.4)$, $\mu_B=(.02,-.01)$, $\mu_P=(.024,-.012)$ and $\Gamma=I_4$, the update yields allocation $0.33$ pp and selection $0.10$ pp, totaling $0.43$ pp against the observed $0.46$ pp. The resulting $0.03$ pp gap shows that the paper's claimed $0.13$ pp selection does not follow from that single restriction alone. The implementation exposes the gap.

`brinson_interaction_reconciled(facts, covariance)` is an explicit extension: it adds two restrictions preserving the original side returns, then checks closure. The three restrictions are

$$
[-d^\mathsf T,d^\mathsf T]m^*=0,\quad
[w_B^\mathsf T,0]m^*=w_B^\mathsf T\mu_B,\quad
[0,w_P^\mathsf T]m^*=w_P^\mathsf T\mu_P.
$$

This preserves each return and removes the **aggregate** interaction; individual group interaction rows can remain nonzero. It is an extension of the paper, not a reproduction of its single-constraint specification. It requires a covariance capable of satisfying all three constraints; otherwise the function raises.

```python
from brinsons import brinson_interaction_restricted, brinson_interaction_reconciled

facts = pl.DataFrame({"period": [1, 1], "segment": ["one", "two"],
    "portfolio_weight": [.6, .4], "benchmark_weight": [.5, .5],
    "portfolio_return": [.024, -.012], "benchmark_return": [.02, -.01]})
keys = ["B:one", "B:two", "P:one", "P:two"]
cov = pl.DataFrame([{"row": a, "column": b, "covariance": float(a == b)}
                    for a in keys for b in keys])
paper = brinson_interaction_restricted(facts, cov)
exact = brinson_interaction_reconciled(facts, cov)
```

### Group covariance propagation

`brinson_group_covariance(assets, residual_covariance)` implements equations (24)–(26). The asset frame has unique `asset`, `segment`, `portfolio_weight`, and `benchmark_weight`; both sides must have nonzero total weight within each group. If $A_B$ and $A_P$ are within-group normalized weight matrices, and $A=[A_B;A_P]$, it returns $\Gamma=A\Delta A^\mathsf T$ in the required long format.

The caller supplies the complete **asset residual covariance** $\Delta$, keyed by asset IDs. The function does not estimate it. To construct it from a supplied raw return covariance $\Sigma$, call `benchmark_residual_covariance(assets, return_covariance)`: it calculates the paper's $\Delta=M_B\Sigma M_B^\mathsf T$ using the benchmark regression residual-maker $M_B$. Since $A_BM_B=0$, this construction gives a zero benchmark covariance block in exact arithmetic. Such a metric can be singular and differs from the positive block-diagonal covariance in the balanced-precision example. The direct Polars joins use substantial memory for large asset universes.

## Precision-weighted multi-period linking

`link_precision_weighted(result, covariance, baseline=None)` implements equations (33)–(35). It takes a `bhb` or `fachler` result and complete period covariance $V$, keyed by period values. Optional `baseline` has unique `period` and `baseline_weight` columns and defaults to ones. For active period returns $e$, compounded active return $E$, and baseline $\bar w$,

$$w^*=\bar w+Ve\,(e^\mathsf TVe)^{-1}(E-e^\mathsf T\bar w),\qquad X_j=\sum_t w_t^*x_{j,t}.$$

The returned detail includes period factors and linked segment effects; its summary reconciles to compounded arithmetic active return. The caller supplies the metric, including any cross-period covariances. The paper suggests diagonal entries $V_{tt}=\mathbf1^\mathsf T\Omega_t\mathbf1$ when $\Omega_t$ is the covariance of the attribution entries in period $t$.

## Entry-specific reconciliation

`link_entry_reconciled(result, covariance)` applies equation (37) to individual period/segment/effect entries:

$$c^*=c+\Sigma\mathbf1\,(\mathbf1^\mathsf T\Sigma\mathbf1)^{-1}(E-\mathbf1^\mathsf Tc).$$

Covariance keys are integer `entry_id` values `0,1,...`, ordered by ascending period, segment, and then `allocation`, `selection`, `interaction`. The returned long detail includes these IDs, `effect`, `estimate`, `adjustment`, and `adjusted`; its summary sums adjusted values by effect. A zero original entry can receive an adjustment. Covariance storage grows quadratically with the number of entries, so the direct implementation is intended for modest inputs.

```python
from brinsons import fachler, link_entry_reconciled
result = fachler(facts_for_several_periods)
n = 3 * result.detail.height
cov = pl.DataFrame([{"row": i, "column": j, "covariance": float(i == j)}
                    for i in range(n) for j in range(n)])
linked = link_entry_reconciled(result, cov)
```

Both linking methods require two or more periods, matching detail and summary, and period returns greater than −100%. They do not estimate covariances automatically.

## Verification and limits

The Hentschel tests cover the paper's numerical example and its reconciliation gap, preserved-return constraints, factor closure and rank failure, residual and group covariance propagation, diagonal and cross-period covariance, nonuniform baselines, entry reconciliation, invalid covariance, and degenerate cases. The paper's sample code uses other choices such as ridge regularization; no direct numerical comparison with it is claimed. The implementation does not estimate or shrink risk models or implement the optional multi-group-constraint appendix.
