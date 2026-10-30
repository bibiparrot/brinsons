"""Write a deterministic, wholly simulated three-account holdings example.

Run from any directory: python scripts/generate_multi_account_data.py
"""
from pathlib import Path

import polars as pl


ACCOUNTS = ("Growth", "Income", "Global")
SECTORS = {
    "Technology": ("TECH_A", "TECH_B"),
    "Healthcare": ("HLTH_A", "HLTH_B"),
}
PERIODS = ("2026-01", "2026-02", "2026-03")
PORTFOLIO_ACCOUNT_WEIGHTS = (
    (.50, .30, .20),
    (.45, .35, .20),
    (.40, .35, .25),
)
BENCHMARK_ACCOUNT_WEIGHTS = (.40, .35, .25)
PORTFOLIO_TECH_WEIGHTS = (.70, .42, .58)
BENCHMARK_TECH_WEIGHTS = (.55, .45, .50)
BENCHMARK_RETURNS = (
    ((.040, .025), (.012, .018)),
    ((-.020, -.012), (.020, .014)),
    ((.032, .024), (-.005, .008)),
)
ACCOUNT_ALPHA = (
    (.004, -.002),
    (-.001, .003),
    (.002, .001),
)


def generate() -> pl.DataFrame:
    """Return 36 unique period/account/sector/security observations."""
    rows = []
    sector_names = tuple(SECTORS)
    for t, period in enumerate(PERIODS):
        for a, account in enumerate(ACCOUNTS):
            p_account = PORTFOLIO_ACCOUNT_WEIGHTS[t][a]
            b_account = BENCHMARK_ACCOUNT_WEIGHTS[a]
            p_tech = PORTFOLIO_TECH_WEIGHTS[a] + (t - 1) * .02 * (1 if a != 1 else -1)
            b_tech = BENCHMARK_TECH_WEIGHTS[a]
            for s, sector in enumerate(sector_names):
                p_sector = p_tech if s == 0 else 1 - p_tech
                b_sector = b_tech if s == 0 else 1 - b_tech
                p_first = .58 + .03 * ((a + t + s) % 3 - 1)
                b_first = .50 + .02 * (s - a)
                for k, security in enumerate(SECTORS[sector]):
                    p_stock = p_first if k == 0 else 1 - p_first
                    b_stock = b_first if k == 0 else 1 - b_first
                    benchmark_return = BENCHMARK_RETURNS[t][s][k]
                    portfolio_return = benchmark_return + ACCOUNT_ALPHA[a][s] + (
                        .001 if k == 0 else -.0005)
                    rows.append({
                        "period": period, "account": account,
                        "sector": sector, "security": security,
                        "portfolio_value": 10_000_000 * p_account * p_sector * p_stock,
                        "benchmark_value": 10_000_000 * b_account * b_sector * b_stock,
                        "portfolio_return": portfolio_return,
                        "benchmark_return": benchmark_return,
                    })
    return pl.DataFrame(rows).sort("period", "account", "sector", "security")


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "data" / "multi_account"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "positions.csv"
    generate().write_csv(path)
    print(f"Wrote {path}")
