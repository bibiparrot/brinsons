# Hierarchical attribution

`roll_up` groups already calculated leaf effects into named ancestors. If $D(h)$ is the set of distinct leaves under parent $h$, it reports

$$X_{h,t}=\sum_{i\in D(h)}X_{i,t},\qquad X\in\{A,S,I\}.$$

The output contains the period key, segment key and selected effect columns only. Pass `effects=[...]` to roll up a different set of numeric effect fields. The inputs are a detail frame and a child/parent relationship frame; `column_child` and `column_parent` select its field names. Parent rows and their descendants are alternative views of the same effects. Summing both double-counts. Recalculating BF on aggregated parent returns can reassign allocation and selection; roll-up deliberately preserves the original leaf decomposition.

For the separate market/currency premium model, use `roll_up_currency_premiums` as described in [Currency attribution](currency.md).
