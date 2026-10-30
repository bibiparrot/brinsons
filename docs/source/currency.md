# Currency attribution

Two separate interfaces serve different currency models. They have different return conventions and should not be combined without an explicit bridge.

## Additive local/currency split (`currency_attribution`)

This function follows the simplified four-effect convention in the local `pybrinson` reference. For each segment, the input supplies simple base returns $r_{P,i}$ and $r_{B,i}$, portfolio local return $\ell_{P,i}$, and one shared currency contribution $c_i$ such that **exactly** $r_{P,i}=\ell_{P,i}+c_i$. The code derives $\ell_{B,i}=r_{B,i}-c_i$. Let $\bar\ell_B=\sum_iw_{B,i}\ell_{B,i}$ and $\bar c_B=\sum_iw_{B,i}c_i$. Then

$$
\begin{aligned}
M_i&=\Delta w_i(\ell_{B,i}-\bar\ell_B),&&\text{market allocation},\\
S_i&=w_{B,i}(\ell_{P,i}-\ell_{B,i}),&&\text{security selection},\\
C_i&=\Delta w_i(c_i-\bar c_B),&&\text{currency allocation},\\
I_i&=\Delta w_i(\ell_{P,i}-\ell_{B,i}),&&\text{interaction}.
\end{aligned}
$$

With normalized weights, $\sum_i(M_i+S_i+C_i+I_i)=R_P-R_B$. The fields are `market_allocation`, `security_selection`, `currency_allocation`, and `interaction`.

This convention is **additive**. For actual simple local and exchange-rate returns, $1+r^{\mathrm{base}}=(1+r^L)(1+c)$, which includes the cross term $r^Lc$. Do not pass raw FX returns as $c_i$ unless the additive equality holds. The function does not model separate currency exposures or hedges.

## Independent market and currency exposures (`currency_premiums`)

This function uses separate market and currency frames. In a market row $i$, let $m_{P,i}$ and $m_{B,i}$ be simple local returns and $h_i$ the simple local cash return. Define log excess over local cash

$$u_{P,i}=\log(1+m_{P,i})-\log(1+h_i),\qquad
u_{B,i}=\log(1+m_{B,i})-\log(1+h_i).$$

Let $\bar u_B=\sum_iw^M_{B,i}u_{B,i}$. In a currency row $c$, define $v_{P,c}=\log(1+f_{P,c})$ and $v_{B,c}=\log(1+f_{B,c})$, with $\bar v_B=\sum_cw^C_{B,c}v_{B,c}$. Market and currency weight vectors are each normalized separately. The four row effects are

$$
\begin{aligned}
M_i&=(w^M_{P,i}-w^M_{B,i})(u_{B,i}-\bar u_B),\\
S_i&=w^M_{P,i}(u_{P,i}-u_{B,i}),\\
C_c&=(w^C_{P,c}-w^C_{B,c})(v_{B,c}-\bar v_B),\\
H_c&=w^C_{P,c}(v_{P,c}-v_{B,c}).
\end{aligned}
$$

The output summary satisfies

$$
L_P=\sum_iw^M_{P,i}u_{P,i}+\sum_cw^C_{P,c}v_{P,c},\quad
L_B=\sum_iw^M_{B,i}u_{B,i}+\sum_cw^C_{B,c}v_{B,c},\quad
\sum_i(M_i+S_i)+\sum_c(C_c+H_c)=L_P-L_B.
$$

`currency_premiums` returns `(market_detail, currency_detail, summary)`. Its summary is a **weighted log-premium model**; $L_P$ is not generally $\log(1+\text{portfolio simple return})$. It resembles Karnosky–Singer's market/currency premium separation but is not a complete implementation of every forward, hedge, and interest-rate convention in the original monograph.

## Aggregating the premium grid (`roll_up_currency_premiums`)

The helper sums each effect by its own market or currency identifier, and computes cumulative log effects by period:

$$X_{k,\le t}=\sum_{s\le t}X_{k,s},\qquad
L_{P,\le t}=\sum_{s\le t}L_{P,s},\quad
L_{B,\le t}=\sum_{s\le t}L_{B,s}.$$

It returns `(market_totals, currency_totals, cumulative_summary)`. Its additive cumulative log values are **not** arithmetic compounding of simple portfolio returns.
