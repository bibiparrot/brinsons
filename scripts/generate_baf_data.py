"""Generate deterministic fictional daily holdings for the BAF tutorials."""

from datetime import date, timedelta
from math import sin
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "baf" / "synthetic_daily_positions.csv"


def business_dates(start: date, count: int) -> list[date]:
    dates = []
    day = start
    while len(dates) < count:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return dates


def episode_shares(day: int, start: int, scale: float) -> float:
    step = day - start
    if step < 0 or step >= 45:
        return 0.0
    if step < 5:
        return 40.0 * scale
    if step < 10:
        return 70.0 * scale
    if step < 20:
        return 100.0 * scale
    if step < 25:
        return 92.0 * scale
    if step < 35:
        return 100.0 * scale
    if step < 40:
        return 70.0 * scale
    return 40.0 * scale


def main() -> None:
    dates = business_dates(date(2025, 1, 2), 180)
    rows = []
    for p in range(3):
        benchmark = 100.0
        prices = [75.0 + 7.0 * s + 4.0 * p for s in range(8)]
        for i, day in enumerate(dates):
            benchmark *= 1 + 0.0004 + 0.003 * sin(i * 0.14 + p * 0.3)
            for s in range(8):
                prices[s] *= 1 + 0.0003 + 0.0015 * (s - 3.5) / 3.5 + 0.006 * sin(i * 0.17 + s * 0.9 + p * 0.6)
                shares = (episode_shares(i, 4 + s % 4, 1 + 0.05 * s) +
                          episode_shares(i, 80 + (s + p) % 5, 0.9 + 0.08 * p))
                rows.append({"portfolio": f"Manager {p + 1}", "date": day,
                             "security": f"FICTIONAL_{s + 1}", "shares": shares,
                             "price": round(prices[s], 8), "benchmark_level": round(benchmark, 8)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT)
    print(f"Wrote {len(rows)} fictional rows to {OUT}")


if __name__ == "__main__":
    main()
