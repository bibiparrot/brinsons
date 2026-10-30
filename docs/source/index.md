# brinsons

**Portfolio attribution with Polars expressions.** Start with a single period, choose a named linking method for multiple periods, then inspect the effects by segment or in a report. Returns, weights and effects use decimal units: `0.01` is 1% or 100 basis points.

```{toctree}
:maxdepth: 2
:caption: Start here

quickstart
data-contract
tutorials
```

```{toctree}
:maxdepth: 2
:caption: Methods and equations

single-period
linking
currency
hierarchy
multi-account
hentschel
behavioral-alpha-frontier
```

```{toctree}
:maxdepth: 2
:caption: Use and verify

reporting
api
verification
references
```

The implementation uses Polars as its only core runtime dependency. The [method-to-code matrix](verification.md) states exactly which equations each function implements and how they were checked. The source checkout also contains extended research notes under `docs/papers`.
