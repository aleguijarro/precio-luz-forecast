"""Direct multi-horizon gradient boosting with quantile outputs, plus naive baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES, build_features, extend_to

QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class HorizonModel:
    k: int
    models: dict[float, HistGradientBoostingRegressor]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {q: m.predict(X[FEATURES]) for q, m in self.models.items()}
        out = pd.DataFrame(preds, index=X.index)
        # Quantile models are trained independently; enforce p10 <= p50 <= p90.
        arr = np.sort(out[list(QUANTILES)].to_numpy(), axis=1)
        return pd.DataFrame(arr, index=X.index, columns=["p10", "p50", "p90"])


def _make_model(q: float, seed: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=[FEATURES.index("hour"), FEATURES.index("dow")],
        random_state=seed,
    )


def training_rows(matrix: pd.DataFrame, k: int) -> pd.DataFrame:
    feats = build_features(matrix, k)
    return feats.dropna(subset=["y", "lag_h", "lag_mean"])


def fit(train: pd.DataFrame, k: int) -> HorizonModel:
    models = {}
    for q in QUANTILES:
        m = _make_model(q)
        m.fit(train[FEATURES], train["y"])
        models[q] = m
    return HorizonModel(k=k, models=models)


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def backtest(matrix: pd.DataFrame, k: int, test_days: int = 60) -> dict:
    """Hold out the last ``test_days`` days: train on everything before, score on the rest.

    Compares the model against two baselines a sensible analyst would use:
    * persistence: the price at the same hour k days earlier;
    * seasonal persistence: the price at the same hour one week earlier.
    """
    rows = training_rows(matrix, k)
    cutoff = rows["day"].max() - pd.Timedelta(days=test_days - 1)
    train, test = rows[rows["day"] < cutoff], rows[rows["day"] >= cutoff].copy()
    test = test.dropna(subset=["week_h"])
    model = fit(train, k)
    pred = model.predict(test)
    y = test["y"].to_numpy()
    res = {
        "k": k,
        "test_days": int(test["day"].nunique()),
        "test_start": cutoff.date().isoformat(),
        "model": _scores(y, pred["p50"].to_numpy()),
        "persistence": _scores(y, test["lag_h"].to_numpy()),
        "seasonal_persistence": _scores(y, test["week_h"].to_numpy()),
        "interval_coverage_80": float(np.mean((y >= pred["p10"]) & (y <= pred["p90"]))),
    }
    best_baseline = min(res["persistence"]["mae"], res["seasonal_persistence"]["mae"])
    res["skill_vs_best_baseline"] = float(1 - res["model"]["mae"] / best_baseline)
    return res


def forecast_day(model: HorizonModel, matrix: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    """Hourly p10/p50/p90 for ``target`` using only data up to ``target - k``."""
    ext = extend_to(matrix, target)
    feats = build_features(ext, model.k)
    rows = feats[feats["day"] == target]
    if rows.empty or rows["lag_h"].isna().any():
        raise ValueError(f"Not enough history to forecast {target.date()} with k={model.k}")
    pred = model.predict(rows)
    pred.insert(0, "hour", rows["hour"].to_numpy())
    return pred.reset_index(drop=True)
