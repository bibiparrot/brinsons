# Brinson performance attribution: single and multiple periods

This note synthesizes the local `pybrinson` source and guides, the papers under `docs/`, and the cited literature. It focuses on return attribution relative to a benchmark, with formulas written in a consistent notation. The equations describe decision contributions, not statistical proof of causation or risk-adjusted skill.

## 1. Setup and assumptions

For segment $i=1,\ldots,N$ and period $t$, let $w_{P,i,t}$ and $w_{B,i,t}$ be portfolio and benchmark weights, and $r_{P,i,t}$ and $r_{B,i,t}$ their segment returns. For a single period suppress $t$. With aligned, exhaustive segments and weights summing to one on each side,

$$
R_P=\sum_i w_{P,i}r_{P,i},\qquad R_B=\sum_i w_{B,i}r_{B,i},\qquad e=R_P-R_B.
$$

Set $\Delta w_i=w_{P,i}-w_{B,i}$ and $\Delta r_i=r_{P,i}-r_{B,i}$. Returns and effects are decimal returns; multiply by 100 for percentage points. These formulas assume weights and segment returns are defined consistently for the period, generally beginning weights and returns over the same period. Trading, external cash flows, fees, derivatives and short exposures require a consistent contribution and return methodology; exact closure alone does not validate those inputs. An empty portfolio or benchmark segment has an undefined conditional return, requiring a documented convention rather than a silent arbitrary value.

## 2. Single-period arithmetic attribution

### Brinson–Hood–Beebower (BHB)

The three effects for each segment are

$$
\begin{aligned}
A_i^{\mathrm{BHB}}&=\Delta w_i\,r_{B,i},\\
S_i&=w_{B,i}\,\Delta r_i,\\
I_i&=\Delta w_i\,\Delta r_i.
\end{aligned}
$$

Allocation $A_i$ isolates the active segment weight at the benchmark segment return; selection $S_i$ isolates the segment return difference at the benchmark weight; interaction $I_i$ captures the joint deviation. For $X\in\{A,S,I\}$, $X=\sum_i X_i$, and

$$
A+S+I=\sum_i\bigl(w_{P,i}r_{P,i}-w_{B,i}r_{B,i}\bigr)=R_P-R_B.
$$

BHB allocation is relative to a zero return at segment level, so its sign should not be read as whether the segment beat the overall benchmark. The BHB framework and interpretation of policy versus active decisions originate in [Brinson, Hood and Beebower (1986)](https://doi.org/10.2469/faj.v42.n4.39); this note uses the explicit three-term convention implemented by pybrinson.

### Brinson–Fachler (BF)

BF changes the allocation reference to the total benchmark return; selection and interaction stay the same:

$$
A_i^{\mathrm{BF}}=\Delta w_i(r_{B,i}-R_B),\qquad
S_i=w_{B,i}\Delta r_i,\qquad I_i=\Delta w_i\Delta r_i.
$$

An overweight of a segment outperforming the total benchmark has positive allocation. When both weight vectors sum to one, $\sum_i\Delta w_i=0$, so BHB and BF have identical total allocation and total excess, although the individual allocation rows differ. See [Brinson and Fachler (1985)](https://doi.org/10.3905/jpm.1985.409005).

A common two-effect presentation folds interaction into selection:

$$
S_i^{(2)}=S_i+I_i=w_{P,i}(r_{P,i}-r_{B,i}),\qquad e=\sum_i(A_i+S_i^{(2)}).
$$

This is a reporting convention: the interaction has been reassigned, not eliminated economically. Other interaction assignments are possible.

### Small example

| Segment | $w_P$ | $w_B$ | $r_P$ | $r_B$ | BF allocation | Selection | Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 60% | 50% | 12% | 10% | +0.20 pp | +1.00 pp | +0.20 pp |
| B | 40% | 50% | 4% | 6% | +0.20 pp | −1.00 pp | +0.20 pp |
| Total | 100% | 100% | **8.80%** | **8.00%** | **+0.40 pp** | **0.00 pp** | **+0.40 pp** |

Thus active return is $0.80$ percentage points. In this example the two-effect selection is $+0.40$ pp.

### Hierarchical reporting

For a parent $h$ with leaf descendants $D(h)$, pybrinson rolls up already-computed effects:

$$ X_h=\sum_{i\in D(h)}X_i,\qquad X\in\{A,S,I\}. $$

This preserves leaf attribution and supports drill-down. Do not sum parent rows together with their descendants: that double-counts. Roll-up of leaf BF effects is not in general the same as recalculating a fresh Brinson decomposition on aggregated parent returns: within-parent active weights and segment return differences can change how effects are assigned. See [hierarchy guide](https://github.com/gghez/pybrinson/blob/529b0940937caacec3f2a30609b9ce6b86316a7b/docs/guides/hierarchies.md) and [implementation](../../tests/vendor/pybrinson/_hierarchy.py).

## 3. Currency attribution and its conventions

For simple local asset return $r^L$ and currency return $c$ (the change in base-currency value of one unit of local currency), exact base-currency return is

$$ 1+r^{\mathrm{base}}=(1+r^L)(1+c),\qquad r^{\mathrm{base}}=r^L+c+r^Lc. $$

Thus adding raw simple local and FX returns omits the cross term. Log returns satisfy $\log(1+r^{\mathrm{base}})=\log(1+r^L)+\log(1+c)$, but a weighted average of asset log returns is generally not the log return of the whole portfolio.

**What pybrinson actually implements.** Its four-effect routine assumes a shared additive currency contribution $c_i$ for portfolio and benchmark within each segment, and requires $r_{P,i}=r^L_{P,i}+c_i$; it derives $r^L_{B,i}=r_{B,i}-c_i$. Define $\bar r^L_B=\sum_iw_{B,i}r^L_{B,i}$ and $\bar c_B=\sum_iw_{B,i}c_i$. Then

$$
\begin{aligned}
M_i&=\Delta w_i(r^L_{B,i}-\bar r^L_B),\\
S_i&=w_{B,i}(r^L_{P,i}-r^L_{B,i}),\\
C_i&=\Delta w_i(c_i-\bar c_B),\\
I_i&=\Delta w_i(r^L_{P,i}-r^L_{B,i}),\\
R_P-R_B&=\sum_i(M_i+S_i+C_i+I_i).
\end{aligned}
$$

These are market allocation, local selection, currency allocation and interaction. They close under the stated additive input convention; they do not generally cover independent currency exposures, hedges, or explicit forward-premium and currency-surprise attribution. The original [Karnosky–Singer framework](https://rpc.cfainstitute.org/research/foundation/1994/global-asset-management-and-performance-attribution) addresses currency and local-market return premiums; [Bacon (2019)](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-lit-review/2019/rflr-performance-attribution.pdf) discusses its treatment of local interest rates and currency. Treat the package routine as a simplified Karnosky–Singer-inspired decomposition, not the full framework. See the [currency source](../../tests/vendor/pybrinson/single_period/currency.py).

## 4. Why multiple periods require linking

For periods $t=1,\ldots,T$, define per-period effects $x_{j,t}$ for effect or segment channel $j$, with $\sum_jx_{j,t}=e_t=R_{P,t}-R_{B,t}$. Define cumulative returns and arithmetic excess:

$$
R_P^{(T)}=\prod_{t=1}^{T}(1+R_{P,t})-1,\quad
R_B^{(T)}=\prod_{t=1}^{T}(1+R_{B,t})-1,\quad
E=R_P^{(T)}-R_B^{(T)}.
$$

Generally $E\ne\sum_t e_t$. Arithmetic linkers seek $X_j=\sum_t\beta_t x_{j,t}$ such that $\sum_jX_j=E$. Closure does not uniquely determine the $\beta_t$: different methods allocate compounding across channels differently. In contrast, geometric active return is

$$ G=\frac{1+R_P^{(T)}}{1+R_B^{(T)}}-1, $$

which differs from $E$ and requires a nonzero benchmark wealth factor. [Bacon's review](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-lit-review/2019/rflr-performance-attribution.pdf) distinguishes smoothing approaches from path-based linking.

For example, portfolio returns $10\%,-5\%$ and benchmark returns $5\%,2\%$ yield $R_P^{(2)}=4.50\%$, $R_B^{(2)}=7.10\%$ and $E=-2.60$ pp. The unlinked active returns sum to $5-7=-2.00$ pp. GRAP coefficients are $1.02$ and $1.10$, giving $1.02(5\%)+1.10(-7\%)=-2.60$ pp.

### Cariño logarithmic smoothing

Let

$$
k(a,b)=\begin{cases}
\dfrac{\ln(1+a)-\ln(1+b)}{a-b},&a\ne b,\\[4pt]
\dfrac{1}{1+b},&a=b,
\end{cases}\quad
k_t=k(R_{P,t},R_{B,t}),\quad K=k(R_P^{(T)},R_B^{(T)}).
$$

Then

$$ X_j^{\mathrm{Carino}}=\sum_t \frac{k_t}{K}x_{j,t}. $$

Since $k_t e_t$ is the period log-relative return, these terms sum to $K E$; division by $K$ closes arithmetic excess. Positive wealth factors are required for the logs. Source: Cariño (1999), as cited in [Bacon (2019)](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-lit-review/2019/rflr-performance-attribution.pdf); the original article was not accessible for direct inspection here. See [implementation](../../tests/vendor/pybrinson/linking/carino.py).

### GRAP factor linking

With empty products equal to one,

$$
\beta_t^{\mathrm{GRAP}}=\left[\prod_{s<t}(1+R_{P,s})\right]
\left[\prod_{s>t}(1+R_{B,s})\right],\qquad
X_j^{\mathrm{GRAP}}=\sum_t\beta_t^{\mathrm{GRAP}}x_{j,t}.
$$

The factors telescope to $E$. Prior portfolio growth scales the period effect; later benchmark growth carries it to the end. See [GRAP implementation](../../tests/vendor/pybrinson/linking/grap.py). The 1997 original was not available for direct inspection; [Bacon's literature review](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-lit-review/2019/rflr-performance-attribution.pdf) discusses GRAP, and the local source provides the formula. **Documentation discrepancy:** `docs/methods.md` in pybrinson gives a different product; the equation above follows its actual source and telescoping identity.

### Frongello recursive linking

Let $P_{t-1}=\prod_{s<t}(1+R_{P,s})$ and $L_{j,0}=0$. The recursive implementation is

$$
L_{j,t}=(1+R_{B,t})L_{j,t-1}+P_{t-1}x_{j,t},\qquad X_j^{\mathrm{Frongello}}=L_{j,T}.
$$

Unrolling this recurrence yields exactly the GRAP factor above for each channel. Therefore the two functions in this pybrinson version give the same effects up to floating-point error, although they express the calculation differently. Other published Frongello variants need not have that same equivalence. The local implementation references [Frongello (2002), “Attribution Linking: Proofed and Clarified”](https://frongello.com/support/Works/JPMFall2002.pdf) and “Linking Single Period Attribution Results”; the author PDF could not be fetched directly during this review. See [source](../../tests/vendor/pybrinson/linking/frongello.py).

### Menchero optimized smoothing

For $e_t=R_{P,t}-R_{B,t}$, define

$$
M=\frac{E/T}{(1+R_P^{(T)})^{1/T}-(1+R_B^{(T)})^{1/T}},\quad
D=E-M\sum_t e_t,\quad
\alpha_t=\frac{D e_t}{\sum_s e_s^2},
$$

$$ X_j^{\mathrm{Menchero}}=\sum_t(M+\alpha_t)x_{j,t}. $$

The correction has $\sum_t\alpha_t e_t=D$ and is the minimum-squared-norm adjustment to the common factor $M$ subject to that constraint. When cumulative returns coincide, use the limit $M=(1+R_B^{(T)})^{(T-1)/T}$, **but still calculate** $\alpha_t$ using the residual formula whenever $\sum e_t^2>0$. If every $e_t=0$, take $\alpha_t=0$. See [Menchero (2004)](https://doi.org/10.2469/faj.v60.n4.2638), Menchero (2000), and [pybrinson source](../../tests/vendor/pybrinson/linking/menchero.py).

**Implementation caveat:** the local pybrinson branch for near-zero $E$ sets every $\alpha_t$ to zero, which can fail closure when the period excesses do not sum to zero. For example, portfolio returns $10\%,-10\%$ and benchmark returns $0\%,-1\%$ both compound to $-1\%$, but period excesses sum to $1$ pp. Its identity check raises rather than silently producing a valid Menchero result. This is a code limitation, not a property of the formula.

### Geometric linking

The package transforms the aggregate arithmetic effects using a semi-notional return $R_{s,t}=R_{B,t}+A_t$. For BHB/BF total allocation this equals $\sum_iw_{P,i,t}r_{B,i,t}$ under normalized weights. Define

$$
a_t=\frac{A_t}{1+R_{B,t}},\qquad
s_t=\frac{S_t+I_t}{1+R_{B,t}+A_t}.
$$

Then

$$
1+A_G=\prod_t(1+a_t),\qquad 1+S_G=\prod_t(1+s_t),\qquad
(1+A_G)(1+S_G)-1=G.
$$

The published fields are geometric allocation and geometric selection; arithmetic interaction is incorporated into selection. Their simple sum is generally **not** $G$: $G=A_G+S_G+A_GS_G$. Geometric methods have multiple possible conventions, so this is the package's two-effect construction, not the only geometric attribution. Its implementation requires positive portfolio, benchmark, and semi-notional wealth factors. See [geometric implementation](../../tests/vendor/pybrinson/linking/geometric.py) and Bacon (2008), publisher information [here](https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289).

## 5. Comparison and implementation boundaries

| Method | Result | Main idea | Order sensitivity and constraints |
|---|---|---|---|
| BHB | Single-period arithmetic | Benchmark segment return for allocation | Normalized weights; selection and interaction separate |
| BF | Single-period arithmetic | Segment benchmark return relative to total benchmark | Same aggregate totals as BHB under normalized weights |
| Cariño | Arithmetic, multiperiod | Log-based smoothing | Factors depend on returns, not their sequence; positive wealth factors |
| GRAP / Frongello | Arithmetic, multiperiod | Prior portfolio and later benchmark compounding | Chronological sequence matters; same channel results in this implementation |
| Menchero | Arithmetic, multiperiod | Common factor plus optimized residual correction | Symmetric under reordering whole periods; package has equal-total edge case above |
| Geometric | Multiplicative relative return | Compound geometric allocation and selection | Interaction folded into selection; positive wealth factors required by package |
| Hierarchical roll-up | Reporting structure | Sum leaf effects into parent rows | Not an independent single-period attribution model |
| Currency routine | Single-period arithmetic | Additive local/currency split | Shared currency contribution; no explicit forward or hedge attribution |

The local linkers operate on aggregate `PeriodAttribution` allocation, selection and interaction, rather than returning a linked series for each segment. The formulas above apply channel by channel when consistent segment effects are separately retained and linked. The currency routine returns a distinct four-effect result, and the ordinary three-effect linking API should not be assumed to link its currency channel. The package requires at least two periods for its linking calls and one consistent single-period method. Its GRAP and Frongello source uses `log1p` to compound returns; despite the linking guide claiming otherwise, their current code cannot safely handle returns at or below −100%. Menchero and geometric also explicitly reject them.

## 6. Context from the local research papers

- [Bacon (2019)](https://rpc.cfainstitute.org/research/foundation/2019/performance-attribution-history-and-progress) surveys the history, geographic and conceptual variants, and the difference between smoothing and path-based linking. It is particularly useful for placing the methods in context.
- A CFA Brinson–Fachler guide and a DolphinDB/Python overview give applied explanations of segment effects and reconciliation.
- Hentschel's *Brinson and Factors* proposes a restricted generalized least-squares framework for allocating interaction, factor residuals and multiperiod reconciliation according to estimated uncertainty; see the [implementation note](hentschel-methods.md).
- Grégoire's *Integrating Risk and Return* connects attribution to risk-adjusted analysis; ordinary Brinson effects alone do not establish that returns compensate for risk.
- Acerbi, Csóka and Herings' *Performance Attribution: the Harsanyi Method* explores an alternative attribution framework. It should not be conflated with the standard Brinson formulas or these linkers.
- An attribution-analysis overview provides introductory context.

## 7. Literature and access notes

Primary references supplied for this review: [Brinson, Hood & Beebower (1986)](https://doi.org/10.2469/faj.v42.n4.39); [Brinson & Fachler (1985)](https://doi.org/10.3905/jpm.1985.409005); Cariño (1999), *Combining Attribution Effects Over Time*; GRAP (1997), *Synthèse des modèles d'attribution de performance*; Frongello (2002), *Linking Single Period Attribution Results* and [*Attribution Linking: Proofed and Clarified*](https://frongello.com/support/Works/JPMFall2002.pdf); Menchero (2000), *An Optimized Approach to Linking Attribution Effects Over Time* and [Menchero (2004)](https://doi.org/10.2469/faj.v60.n4.2638); [Karnosky & Singer (1994)](https://rpc.cfainstitute.org/research/foundation/1994/global-asset-management-and-performance-attribution); [Bacon (2008)](https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289); and [Bacon (2019)](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-lit-review/2019/rflr-performance-attribution.pdf).

The local 2019 Bacon literature review and accessible abstracts were consulted; not every listed paper or the full 2008 book was accessible. Equations for inaccessible originals are therefore cross-checked against local source and available reviews rather than presented as verified quotations from those originals. The user-supplied feature matrix for `ppar` and `fincore` is context, not an independently verified comparison; conclusions about implementations here concern the checked-in pybrinson source.
