# Hentschel restricted attribution

The local paper, Hentschel (2026), *Brinson and Factors: A Unified Framework for Complete Portfolio Return Attribution*, treats attribution reconciliation as a constrained estimation problem. The functions here implement selected equations from that paper; the caller supplies covariance matrices. Exact identities alone do not validate a covariance estimate.

## Common restricted GLS update

Given an estimate $\hat\theta$, a covariance or adjustment metric $\Sigma$, and linear constraints $Q\theta=b$, the code computes

$$
\theta^*=\hat\theta+\Sigma Q^\mathsf T
(Q\Sigma Q^\mathsf T)^{-1}(b-Q\hat\theta).
$$

Covariances are complete long-form Polars frames keyed by `row`, `column`, `covariance` by default. They must be finite, symmetric, and positive semidefinite. A singular constraint system requiring a nonzero adjustment raises `AttributionError`; the implementation does not add a ridge or pseudoinverse.

## Factor attribution (`factor_attribution_restricted`)

For asset-return vector $r$, factor-exposure matrix $X$, positive diagonal regression weights $W$, and active asset weights $w_A=w_P-w_B$, set

$$
\hat f=(X^\mathsf TWX)^{-1}X^\mathsf TWr,\qquad
\Omega=(X^\mathsf TWX)^{-1},\qquad
q=X^\mathsf Tw_A,\qquad b=w_A^\mathsf Tr.
$$

The paper's active-exactness restriction is $q^\mathsf Tf^*=b$. The function implements

$$
f^*=\hat f+\Omega q(q^\mathsf T\Omega q)^{-1}
(b-q^\mathsf T\hat f),\qquad c_k^*=q_kf_k^*.
$$

It returns factor exposure, unrestricted and adjusted factor return, and both contributions. The factors must be full rank and the portfolio and benchmark asset weights must each sum to one. The inverse cross-product is used as the adjustment metric; interpreting it as sampling covariance needs a specified error model.

## Removing aggregate Brinson interaction

`brinson_interaction_restricted` begins with one-period BF attribution. Stack segment returns $m=[r_B;r_P]$, set $d=w_P-w_B$ and $q=[-d;d]$. The aggregate interaction is $I=q^\mathsf Tm$. The paper's single restriction gives

$$m^*=m-\Gamma q(q^\mathsf T\Gamma q)^{-1}I.$$

The resulting aggregate interaction is zero. This single restriction can alter the implied portfolio and benchmark returns. The function retains the **observed** side returns in its summary and exposes `reconciliation_residual = observed excess - adjusted allocation - adjusted selection`. The paper's numerical two-group example has a nonzero residual under the single restriction; see the [verification notes](verification.md).

`brinson_interaction_reconciled` is an explicit extension that also preserves both observed side returns. Its constraint matrix has rows

$$
q^\mathsf Tm^*=0,\qquad
[w_B^\mathsf T,0]m^*=w_B^\mathsf Tr_B,\qquad
[0,w_P^\mathsf T]m^*=w_P^\mathsf Tr_P.
$$

It therefore forces aggregate interaction to zero **and** $A^*+S^*=R_P-R_B$. Individual segment interaction rows can remain nonzero.

## Covariance construction

`benchmark_residual_covariance` uses benchmark within-segment normalized weights to form a residual-maker $M_B=I-D(D^\mathsf TW_BD)^{-1}D^\mathsf TW_B$, where $D$ is the segment indicator matrix. Given an asset return covariance $\Sigma_r$, it returns the paper's equation (23):

$$\Delta=M_B\Sigma_r M_B^\mathsf T.$$

`brinson_group_covariance` receives an asset **residual** covariance $\Delta$ and within-segment weight aggregation matrices $A_B,A_P$. With $A=[A_B;A_P]$, it produces equations (24)–(26):

$$\Gamma=A\Delta A^\mathsf T.$$

The benchmark residual construction can make the benchmark block of $\Gamma$ zero and the full matrix singular. These functions do not estimate $\Sigma_r$ from returns.

## Period precision linking (`link_precision_weighted`)

For period active returns $g=(g_t)$, compounded arithmetic excess $E$, baseline linking weights $\bar\beta$ (ones by default), and period metric $V$, the paper's equation (33) gives

$$
\beta^*=\bar\beta+Vg(g^\mathsf TVg)^{-1}
(E-g^\mathsf T\bar\beta),\qquad
X_j=\sum_t\beta_t^*x_{j,t}.
$$

The optional `baseline` frame supplies period and `baseline_weight` values. The paper suggests $V_{tt}=\mathbf1^\mathsf T\Omega_t\mathbf1$ for a diagonal period metric; the function accepts a supplied full period covariance. It requires at least two periods.

## Entry-specific reconciliation (`link_entry_reconciled`)

Stack every period/segment/effect estimate into $c$ and supply an entry covariance $\Sigma_c$. Equation (37) adjusts each entry directly:

$$
c^*=c+\Sigma_c\mathbf1
(\mathbf1^\mathsf T\Sigma_c\mathbf1)^{-1}
(E-\mathbf1^\mathsf Tc).
$$

The output includes `entry_id`, `effect`, `estimate`, `adjustment`, and `adjusted`. IDs follow sorted period, segment, then allocation/selection/interaction order. Zero original entries may receive a nonzero adjustment. Covariance storage grows quadratically in the entry count; this direct method is intended for modest inputs.
