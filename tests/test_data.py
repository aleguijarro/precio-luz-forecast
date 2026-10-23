import pandas as pd

from luzcast.data import merge_prices, parse_response


def _payload():
    return {
        "data": {"type": "Precios mercado peninsular en tiempo real"},
        "included": [
            {
                "type": "PVPC",
                "id": "1001",
                "attributes": {
                    "values": [
                        {"value": 208.94, "percentage": 0.47, "datetime": "2026-10-20T00:00:00.000+02:00"},
                        {"value": 200.97, "percentage": 0.47, "datetime": "2026-10-20T01:00:00.000+02:00"},
                    ]
                },
            },
            {
                "type": "Precio mercado spot",
                "id": "600",
                "attributes": {"values": [{"value": 235, "datetime": "2026-10-20T00:00:00.000+02:00"}]},
            },
        ],
    }


def test_parse_picks_pvpc_series_and_converts_to_utc():
    df = parse_response(_payload())
    assert list(df.columns) == ["datetime", "pvpc"]
    assert len(df) == 2
    assert df["datetime"].iloc[0] == pd.Timestamp("2026-10-19T22:00Z")
    assert df["pvpc"].iloc[0] == 208.94


def test_parse_missing_series_returns_empty():
    assert parse_response({"included": []}).empty


def test_merge_prefers_newest_value():
    t = pd.Timestamp("2026-10-20T10:00Z")
    old = pd.DataFrame({"datetime": [t], "pvpc": [100.0]})
    new = pd.DataFrame({"datetime": [t, t + pd.Timedelta(hours=1)], "pvpc": [110.0, 120.0]})
    merged = merge_prices(old, new)
    assert merged["pvpc"].tolist() == [110.0, 120.0]
