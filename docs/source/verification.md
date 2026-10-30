# Formula and implementation checks

The documented equations were compared against the Polars expressions in `src/brinsons/core.py` and `src/brinsons/hentschel.py`. The Hentschel equations were also checked against the local 40-page paper, in particular its equations (3), (14), (20), (23), (33) and (37). `tests/test_documented_equations.py` uses independent scalar calculations for the main Brinson, linking and currency formulas; `tests/test_hentschel.py` checks the matrix formulas and constraints with small numerical examples.

| Documented method | Function | Verification |
|---|---|---|
| BHB, BF, 2/3 effects | `bhb`, `fachler` | Scalar row equations, summary closure, custom columns |
| Cariño | `link_carino` | Log-slope factors and compounded arithmetic excess |
| GRAP | `link_grap` | Explicit prior-portfolio/later-benchmark factors |
| Frongello | `link_frongello` | Equality to GRAP and recurrence |
| Menchero | `link_menchero` | Scalar optimized factors and equal-total limit |
| Geometric | `link_geometric` | Two compounded effects and multiplicative identity |
| Currency additive | `currency_attribution` | Four row effects and additive input identity |
| Currency premiums | `currency_premiums` | Four log-premium row effects and active log closure |
| Hierarchy | `roll_up` | Sum of distinct leaf effects |
| Restricted factor | `factor_attribution_restricted` | Regression and active-exactness constraint |
| Brinson interaction | `brinson_interaction_restricted`, `brinson_interaction_reconciled` | Paper single restriction, residual, and preserved-return extension |
| Covariance propagation | `benchmark_residual_covariance`, `brinson_group_covariance` | $M\Sigma M^\mathsf T$ and $A\Delta A^\mathsf T$ |
| Precision linking | `link_precision_weighted` | Period GLS factors and closure |
| Entry reconciliation | `link_entry_reconciled` | Entry GLS adjustment and closure |
| Multi-account Macro→Micro | `macro_micro_attribution` | Account selection/interaction replacement by weighted micro effects; direct value-weighted return closure |
| Pooled multi-account | `value_weighted_total_attribution` | Matches direct total portfolio and weighted benchmark returns |
| Decision hierarchy | `hierarchical_decision_attribution` | Independent Python successive-notional values $N_0$–$N_4$ and each stage difference |
| Multi-account linking | `link_macro_micro_attribution`, `link_hierarchical_decision_attribution` | Four arithmetic linkers, full-horizon closure, and geometric rejection |
| Behavioral Alpha Frontier | `calculate_baf`, `plot_baf`, `baf_change_table`, `chart_baf_change_table` | Episode separation, seven decision formulas, payoff scaling, curved zero-edge boundary, paired-peer percentage changes, and table colors |

Run the full suite and a warning-as-error documentation build from the repository root:

```console
python -m pip install -e ".[test,docs]"
python -m pytest -q
python -m sphinx -b html -W --keep-going docs/source docs/_build/html
```

The integration suite also compares suitable methods with the [MIT-licensed test reference](../../tests/vendor/README.md) under `tests/vendor`. The tests establish agreement for their fixtures and the stated conventions; they cannot validate external data quality, covariance estimation, or economic interpretation. The paper's single interaction restriction can leave a nonzero `reconciliation_residual` because it does not preserve the original side returns. The package's `brinson_interaction_reconciled` adds two constraints specifically to resolve that difference.
