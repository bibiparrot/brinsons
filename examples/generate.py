"""Generate deterministic sample data and write period and linked results."""
from pathlib import Path
import random
import polars as pl
from brinsons import fachler, link_carino

rng = random.Random(2026)
rows = []
for period in range(1, 13):
    p = [rng.uniform(.1, 1) for _ in range(4)]
    b = [rng.uniform(.1, 1) for _ in range(4)]
    for i, segment in enumerate(["Equity", "Bonds", "Cash", "Other"]):
        rows.append({"period": period, "segment": segment,
            "portfolio_weight": p[i]/sum(p), "benchmark_weight": b[i]/sum(b),
            "portfolio_return": rng.uniform(-.08, .12),
            "benchmark_return": rng.uniform(-.08, .12)})

data = pl.DataFrame(rows)
result = fachler(data)
linked = link_carino(result)
out = Path(__file__).resolve().parents[1] / "data" / "examples"
out.mkdir(parents=True, exist_ok=True)
data.write_csv(out / "input.csv")
result.detail.write_csv(out / "period_detail.csv")
result.summary.write_csv(out / "period_summary.csv")
linked.detail.write_csv(out / "linked_detail.csv")
linked.summary.write_csv(out / "linked_summary.csv")
print(f"Wrote example CSV files to {out}")
