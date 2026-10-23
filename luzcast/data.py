"""Download and store hourly PVPC prices from Red Eléctrica's public API (apidatos.ree.es).

No API key is needed. The endpoint returns a JSON:API document where each series lives in
``included[*].attributes.values`` as ``{"value", "percentage", "datetime"}`` records.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
TZ = "Europe/Madrid"
PVPC_SERIES_ID = "1001"
DEFAULT_START = date(2022, 1, 1)
CSV_COLUMNS = ["datetime", "pvpc"]


def parse_response(payload: dict, series_id: str = PVPC_SERIES_ID) -> pd.DataFrame:
    """Turn an apidatos JSON document into a DataFrame with UTC ``datetime`` and ``pvpc`` (€/MWh)."""
    for item in payload.get("included", []):
        if str(item.get("id")) == series_id:
            values = item["attributes"]["values"]
            break
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.DataFrame(values)
    if df.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(df["datetime"], utc=True),
            "pvpc": pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    # Some periods may come at sub-hourly resolution; normalise to hourly means.
    out["datetime"] = out["datetime"].dt.floor("h")
    out = out.groupby("datetime", as_index=False)["pvpc"].mean()
    return out.dropna()


def fetch_range(start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch PVPC between two local dates (inclusive), in chunks the API accepts (<= 28 days)."""
    session = session or requests.Session()
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=27), end)
        params = {
            "start_date": f"{cursor.isoformat()}T00:00",
            "end_date": f"{chunk_end.isoformat()}T23:59",
            "time_trunc": "hour",
        }
        for attempt in range(4):
            try:
                resp = session.get(API_URL, params=params, timeout=30)
                if resp.status_code == 200:
                    frames.append(parse_response(resp.json()))
                    break
                # 500 usually means "no data yet" for future days; don't retry those forever.
                if resp.status_code >= 500 and chunk_end >= pd.Timestamp.now(tz=TZ).date():
                    break
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def merge_prices(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Combine, preferring the most recent download for duplicated hours."""
    frames = [f for f in (old, new) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("datetime", keep="last").sort_values("datetime")
    return df.reset_index(drop=True)


def update_prices(path: Path, today: date | None = None) -> pd.DataFrame:
    """Incrementally update the CSV: backfill on first run, then refresh the last few days."""
    today = today or pd.Timestamp.now(tz=TZ).date()
    old = load_prices(path)
    if old.empty:
        start = DEFAULT_START
    else:
        last_local = old["datetime"].max().tz_convert(TZ).date()
        start = last_local - timedelta(days=3)  # re-download recent days in case of revisions
    end = today + timedelta(days=1)  # tomorrow's PVPC is published around 20:15 CET
    new = fetch_range(start, end)
    df = merge_prices(old, new)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["pvpc"] = out["pvpc"].round(2)
    out.to_csv(path, index=False)
    return df
