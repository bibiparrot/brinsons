# Single-period Brinson attribution

For one period, set $\Delta w_i=w_{P,i}-w_{B,i}$ and $\Delta r_i=r_{P,i}-r_{B,i}$. The observed side returns and arithmetic excess are

$$
R_P=\sum_i w_{P,i}r_{P,i},\qquad
R_B=\sum_i w_{B,i}r_{B,i},\qquad
e=R_P-R_B.
$$

## Brinson–Hood–Beebower (`bhb`)

With `effects=3`, the segment effects are

$$
\begin{aligned}
A_i^{\mathrm{BHB}}&=\Delta w_i r_{B,i},\\
S_i&=w_{B,i}\Delta r_i,\\
I_i&=\Delta w_i\Delta r_i.
\end{aligned}
$$

The allocation reference is zero return. For each segment,

$$A_i+S_i+I_i=w_{P,i}r_{P,i}-w_{B,i}r_{B,i}.$$

Summing all segments gives $A+S+I=e$.

## Brinson–Fachler (`fachler`)

The allocation reference is the **whole benchmark return**:

$$
A_i^{\mathrm{BF}}=\Delta w_i(r_{B,i}-R_B),\qquad
S_i=w_{B,i}\Delta r_i,\qquad I_i=\Delta w_i\Delta r_i.
$$

The BF segment row differs from its direct contribution difference by $-\Delta w_iR_B$. With normalized weights, $\sum_i\Delta w_i=0$, so the offset cancels in total and $A+S+I=e$. Under the same condition, BHB and BF **aggregate** allocation are equal, while category allocation can differ.

## Two-effect presentation

Both functions accept `effects=2`. Interaction is included in selection:

$$
A_i^{(2)}=A_i^{(3)},\qquad
S_i^{(2)}=w_{P,i}\Delta r_i=S_i^{(3)}+I_i^{(3)},\qquad
I_i^{(2)}=0.
$$

The output keeps an `interaction` field with zeros. This changes the presentation of selection, not the total excess.

## Numeric check

| Segment | $w_P$ | $w_B$ | $r_P$ | $r_B$ | BF $A_i$ | $S_i$ | $I_i$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equity | .60 | .50 | .12 | .10 | .002 | .010 | .002 |
| Bonds | .40 | .50 | .04 | .06 | .002 | −.010 | .002 |
| Total | 1 | 1 | $R_P=.088$ | $R_B=.080$ | .004 | 0 | .004 |

Thus the reported total is $.004+0+.004=.008=R_P-R_B$. The [equation tests](verification.md) check each category and total, including both effect presentations.
