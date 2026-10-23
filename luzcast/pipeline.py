"""End-to-end daily job: update data -> evaluate -> train -> forecast -> write site JSON.

Run with ``python -m luzcast`` (GitHub Actions does this on a schedule).
Pass ``--offline`` to skip the download and use the CSV already in ``data/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import is_holiday
from .data import TZ, load_prices, update_prices
from .features import complete_days, to_daily_matrix
from .model import backtest, fit, forecast_day, training_rows

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "pvpc.csv"
HISTORY_CSV = ROOT / "data" / "forecast_history.csv"
SITE_JSON = ROOT / "site" / "data" / "forecast.json"

MAX_K = 3          # horizons trained: 1, 2 and 3 days ahead
WINDOW_HOURS = 3   # length of the "best window" shown on the site
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def kwh(x: float) -> float:
    """€/MWh -> €/kWh rounded for display."""
    return round(float(x) / 1000, 4)


def spanish_date(d: date) -> str:
    return f"{DIAS[d.weekday()]} {d.day} de {MESES[d.month - 1]}"


def best_window(values: list[float], width: int = WINDOW_HOURS) -> int:
    sums = [sum(values[i : i + width]) for i in range(len(values) - width + 1)]
    return int(np.argmin(sums))


def day_payload(d: date, label: str, hours: pd.DataFrame, source: str, k: int | None) -> dict:
    p50 = hours["p50"].tolist()
    cheapest = sorted(range(24), key=lambda h: p50[h])[:6]
    start = best_window(p50)
    return {
        "date": d.isoformat(),
        "label": label,
        "pretty_date": spanish_date(d),
        "is_offday": bool(d.weekday() >= 5 or is_holiday(d)),
        "source": source,                     # "official" (published by REE) or "forecast"
        "horizon_days": k,
        "hours": [
            {
                "hour": int(r.hour),
                "price": kwh(r.p50),
                "low": kwh(r.p10),
                "high": kwh(r.p90),
            }
            for r in hours.itertuples()
        ],
        "summary": {
            "mean": kwh(np.mean(p50)),
            "min": kwh(min(p50)),
            "min_hour": int(np.argmin(p50)),
            "max": kwh(max(p50)),
            "max_hour": int(np.argmax(p50)),
            "cheapest_hours": sorted(cheapest),
            "best_window_start": start,
            "best_window_end": start + WINDOW_HOURS,
            "best_window_mean": kwh(np.mean(p50[start : start + WINDOW_HOURS])),
        },
    }


def official_hours(matrix: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    row = matrix.loc[d]
    return pd.DataFrame({"hour": range(24), "p10": row.values, "p50": row.values, "p90": row.values})


def update_history(new_rows: pd.DataFrame, matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Append today's forecasts and score every past forecast whose actual price is now known."""
    if HISTORY_CSV.exists():
        hist = pd.read_csv(HISTORY_CSV, parse_dates=["issued", "target_day"])
    else:
        hist = pd.DataFrame(columns=["issued", "target_day", "hour", "k", "p10", "p50", "p90"])
    hist = pd.concat([f for f in (hist, new_rows) if not f.empty] or [hist], ignore_index=True)
    hist["issued"] = pd.to_datetime(hist["issued"])
    hist["target_day"] = pd.to_datetime(hist["target_day"])
    hist[["hour", "k"]] = hist[["hour", "k"]].astype(int)
    hist = hist.drop_duplicates(["issued", "target_day", "hour", "k"], keep="last")
    hist = hist.sort_values(["issued", "target_day", "hour"]).reset_index(drop=True)
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    hist = hist[["issued", "target_day", "hour", "k", "p10", "p50", "p90"]]
    hist[["p10", "p50", "p90"]] = hist[["p10", "p50", "p90"]].astype(float).round(3)
    hist.to_csv(HISTORY_CSV, index=False, date_format="%Y-%m-%d")

    long = matrix.stack().rename("actual").reset_index()
    long.columns = ["target_day", "hour", "actual"]
    long["target_day"] = pd.to_datetime(long["target_day"]).astype(hist["target_day"].dtype)
    long["hour"] = long["hour"].astype(int)
    scored = hist.merge(long, on=["target_day", "hour"], how="inner")
    live = {}
    for k, grp in scored.groupby("k"):
        live[str(int(k))] = {
            "days": int(grp["target_day"].nunique()),
            "mae": round(float(np.mean(np.abs(grp["p50"] - grp["actual"]))) / 1000, 4),
        }
    return hist, live


def run(offline: bool = False, today: date | None = None) -> dict:
    today = today or pd.Timestamp.now(tz=TZ).date()
    prices = load_prices(DATA_CSV) if offline else update_prices(DATA_CSV, today=today)
    if prices.empty:
        raise SystemExit("No price data available")
    matrix = to_daily_matrix(prices)
    known = complete_days(matrix)
    matrix = matrix.loc[: known.max()]
    last_known = known.max()
    print(f"Data: {len(prices):,} hours, last complete day {last_known.date()}")

    metrics, models = {}, {}
    for k in range(1, MAX_K + 1):
        metrics[str(k)] = backtest(matrix, k)
        models[k] = fit(training_rows(matrix, k), k)
        m = metrics[str(k)]
        print(
            f"k={k}: MAE model {m['model']['mae']:.1f} | persistence {m['persistence']['mae']:.1f}"
            f" | weekly {m['seasonal_persistence']['mae']:.1f} €/MWh (skill {m['skill_vs_best_baseline']:+.0%})"
        )

    labels = {0: "Hoy", 1: "Mañana", 2: "Pasado mañana"}
    days, hist_rows = [], []
    for offset, label in labels.items():
        d = pd.Timestamp(today + timedelta(days=offset))
        if d in known:
            days.append(day_payload(d.date(), label, official_hours(matrix, d), "official", None))
            continue
        k = (d - last_known).days
        if not 1 <= k <= MAX_K:
            continue
        hours = forecast_day(models[k], matrix, d)
        days.append(day_payload(d.date(), label, hours, "forecast", k))
        rows = hours.assign(issued=pd.Timestamp(today), target_day=d, k=k)
        hist_rows.append(rows)

    new_hist = pd.concat(hist_rows) if hist_rows else pd.DataFrame()
    _, live = update_history(new_hist, matrix)

    recent = matrix.loc[last_known - pd.Timedelta(days=29) :]
    payload = {
        "generated_at": pd.Timestamp.now(tz=TZ).isoformat(timespec="minutes"),
        "today": today.isoformat(),
        "last_official_day": last_known.date().isoformat(),
        "unit": "€/kWh",
        "days": days,
        "recent_daily": [
            {"date": d.date().isoformat(), "mean": kwh(r.mean()), "min": kwh(r.min()), "max": kwh(r.max())}
            for d, r in recent.iterrows()
        ],
        "backtest": {
            k: {
                "test_days": m["test_days"],
                "mae_model": kwh(m["model"]["mae"]),
                "mae_persistence": kwh(m["persistence"]["mae"]),
                "mae_weekly": kwh(m["seasonal_persistence"]["mae"]),
                "skill": round(m["skill_vs_best_baseline"], 3),
                "coverage_80": round(m["interval_coverage_80"], 3),
            }
            for k, m in metrics.items()
        },
        "live": live,
        "training_days": len(known),
        "first_day": known.min().date().isoformat(),
    }
    SITE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SITE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    (ROOT / "data" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {SITE_JSON.relative_to(ROOT)} with {len(days)} days")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use the existing CSV, no download")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
