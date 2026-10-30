# Multiple-period linking

Let $p_t=R_{P,t}$, $b_t=R_{B,t}$, $g_t=p_t-b_t$, and let $x_{j,t}$ be a period effect for a segment/effect channel $j$, such that $\sum_jx_{j,t}=g_t$. Over $T\ge2$ chronologically ordered periods,

$$
P=\prod_{t=1}^T(1+p_t)-1,\qquad
B=\prod_{t=1}^T(1+b_t)-1,\qquad E=P-B.
$$

Arithmetic linkers below produce $X_j=\sum_t\beta_t x_{j,t}$ and $\sum_jX_j=E$. `result.detail` contains each segment's `linked_allocation`, `linked_selection`, and `linked_interaction`; the linked `summary` has their full-horizon totals. The library requires $p_t,b_t>-1$.

## Cariño (`link_carino`)

Define the logarithmic secant slope

$$
k(a,b)=\begin{cases}
\dfrac{\log(1+a)-\log(1+b)}{a-b},&a\ne b,\\
\dfrac{1}{1+b},&a=b.
\end{cases}
$$

The implementation uses the continuous branch for $|a-b|<10^{-12}$. With $K=k(P,B)$,

$$\beta_t^{\mathrm{Cari\tilde no}}=\frac{k(p_t,b_t)}{K},\qquad X_j=\sum_t\beta_t^{\mathrm{Cari\tilde no}}x_{j,t}.$$

This closes because $\sum_t k(p_t,b_t)g_t=\log(1+P)-\log(1+B)=K E$.

## GRAP (`link_grap`)

With empty products equal to one,

$$
\beta_t^{\mathrm{GRAP}}=
\left[\prod_{s<t}(1+p_s)\right]
\left[\prod_{s>t}(1+b_s)\right],\qquad
X_j=\sum_t\beta_t^{\mathrm{GRAP}}x_{j,t}.
$$

The earlier **portfolio** wealth and later **benchmark** wealth multiply each period effect. The factors telescope, giving $\sum_t\beta_t g_t=E$.

## Frongello (`link_frongello`)

The current function delegates to `link_grap`, so it returns the same linked effects and factors. The equivalent recurrence is

$$
L_{j,0}=0,\qquad
L_{j,t}=(1+b_t)L_{j,t-1}+
\left[\prod_{s<t}(1+p_s)\right]x_{j,t},\qquad X_j=L_{j,T}.
$$

This documents the **implemented** Frongello convention; other presentations of Frongello linking may assign a different recursive correction.

## Menchero (`link_menchero`)

Set $n=T$, $D=E-M\sum_tg_t$, and

$$
M=\begin{cases}
\dfrac{E/T}{(1+P)^{1/T}-(1+B)^{1/T}},&E\ne0,\\[4pt]
(1+B)^{(T-1)/T},&E=0.
\end{cases}
$$

The implemented limit branch is selected at $|E|<10^{-12}$. Then

$$
\alpha_t=\begin{cases}
\dfrac{Dg_t}{\sum_s g_s^2},&\sum_sg_s^2>0,\\[4pt]
0,&\sum_sg_s^2=0,
\end{cases}\qquad
\beta_t^{\mathrm{Menchero}}=M+\alpha_t,\qquad
X_j=\sum_t\beta_t^{\mathrm{Menchero}}x_{j,t}.
$$

The correction is the minimum squared-norm vector satisfying $\sum_t\alpha_tg_t=D$. It is still evaluated when $E=0$ and some $g_t\ne0$.

## Geometric (`link_geometric`)

This method uses a different target, the **compounded relative** return

$$G=\frac{1+P}{1+B}-1.$$

For period *totals* $A_t,S_t,I_t$, let the semi-notional return be $b_t+A_t$, and define

$$
a_t=\frac{A_t}{1+b_t},\qquad
s_t=\frac{S_t+I_t}{1+b_t+A_t}.
$$

The reported horizon effects are

$$
A_G=\prod_t(1+a_t)-1,\qquad
S_G=\prod_t(1+s_t)-1,\qquad
I_G=0,
$$

with **multiplicative** reconciliation

$$G=(1+A_G)(1+S_G)-1=A_G+S_G+A_GS_G.$$

The result's detail contains period-level `geometric_allocation` and `geometric_selection`; it does not contain linked segment effects. The semi-notional wealth $1+b_t+A_t$ must be positive. A geometric waterfall shows the cross term $A_GS_G$ separately.

## Comparing methods

For $p=(.10,-.05)$ and $b=(.05,.02)$, $P=.045$, $B=.071$, and $E=-.026$, whereas $g_1+g_2=-.020$. GRAP gives $\beta=(1.02,1.10)$ and $.05(1.02)-.07(1.10)=-.026$. Cariño and Menchero also sum to $-.026$, but can distribute it differently across effects. GRAP and Frongello are equal in this implementation. Geometric instead targets $(1.045/1.071)-1$.
