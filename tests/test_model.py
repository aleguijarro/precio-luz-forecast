import numpy as np
import pandas as pd
import pytest

from luzcast.features import build_features, complete_days, to_daily_matrix
from luzcast.model import backtest, fit, forecast_day, training_rows


def synthetic_prices(days: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="Europe/Madrid").tz_convert("UTC")
    local = idx.tz_convert("Europe/Madrid")
    hour = local.hour.to_numpy()
    price = 100 + 40 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 5, len(idx))
    return pd.DataFrame({"datetime": idx, "pvpc": price})


def test_matrix_has_24_columns_and_handles_dst():
    m = to_daily_matrix(synthetic_prices())
    assert list(m.columns) == list(range(24))
    # 2025-03-30 is the spring DST change (23 local hours): it must still be complete.
    assert m.loc["2025-03-30"].notna().all()


def test_features_do_not_leak_the_target_day():
    m = to_daily_matrix(synthetic_prices())
    feats = build_features(m, k=2)
    row = feats[(feats["day"] == "2025-05-10") & (feats["hour"] == 9)].iloc[0]
    assert row["lag_h"] == pytest.approx(m.loc["2025-05-08", 9])
    assert row["week_h"] == pytest.approx(m.loc["2025-05-03", 9])
    assert row["y"] == pytest.approx(m.loc["2025-05-10", 9])


def test_model_beats_persistence_on_learnable_signal():
    m = to_daily_matrix(synthetic_prices())
    res = backtest(m, k=1, test_days=30)
    assert res["model"]["mae"] < res["persistence"]["mae"]
    assert 0.5 < res["interval_coverage_80"] <= 1.0


def test_forecast_future_day_returns_ordered_quantiles():
    m = to_daily_matrix(synthetic_prices())
    m = m.loc[: complete_days(m).max()]
    model = fit(training_rows(m, 2), 2)
    target = m.index.max() + pd.Timedelta(days=2)
    pred = forecast_day(model, m, target)
    assert len(pred) == 24
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
