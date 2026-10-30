# Simulated multi-account example

`positions.csv` is deterministic and entirely synthetic. Regenerate it with
`python scripts/generate_multi_account_data.py` from the project root.

Each row is one period/account/sector/security position. `portfolio_value` is
beginning-of-period market value; `benchmark_value` is benchmark notional.
Both totals are 10 million per period. `portfolio_return` and
`benchmark_return` are simple decimal returns over the same period.
Three accounts each hold two sectors and two fictional securities per sector.
The values encode account, sector, and security allocation decisions; return
differences encode simulated selection. No transfers, fees, cash flows, or
real market data are represented.
