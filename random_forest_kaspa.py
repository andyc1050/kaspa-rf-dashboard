"""Walk-forward Random Forest trading research for KASPA daily OHLCV data.

This is a research scaffold, not trading advice. It intentionally avoids
scikit-learn so it can run in the bundled Codex Python runtime with only
pandas and numpy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(r"C:\Users\Andy Chi\Desktop\KASPA_Historical_Data.xlsx")
DEFAULT_CANONICAL_CSV = Path(__file__).resolve().parent / "data" / "kaspa_daily_ohlcv.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FEATURE_LABELS = {
    "close_position": "Close location in daily range",
    "range_pct": "Daily range",
    "atr_14_pct": "14-day ATR",
    "return_2d": "2-day return",
    "return_3d": "3-day return",
    "return_5d": "5-day return",
    "return_30d": "30-day return",
    "return_90d": "90-day return",
    "log_return_1d": "1-day log return",
    "close_vs_sma_5d": "Close vs 5-day SMA",
    "close_vs_sma_20d": "Close vs 20-day SMA",
    "close_vs_sma_50d": "Close vs 50-day SMA",
    "close_vs_sma_200d": "Close vs 200-day SMA",
    "body_pct": "Candle body",
    "volatility_30d": "30-day volatility",
    "volatility_60d": "60-day volatility",
    "volatility_5d": "5-day volatility",
    "volume_ratio_30d": "Volume vs 30-day average",
    "rsi_14": "14-day RSI",
}

FEATURE_NOTES = {
    "close_position": "Shows whether KAS closed near the top or bottom of its daily candle.",
    "range_pct": "Measures intraday movement relative to closing price.",
    "atr_14_pct": "Captures the current volatility regime using average true range.",
    "return_2d": "Short-term momentum input.",
    "return_3d": "Short-term momentum input.",
    "return_5d": "One-week momentum input.",
    "return_30d": "Medium-term trend input.",
    "return_90d": "Quarterly trend input.",
    "log_return_1d": "Latest daily price impulse.",
    "close_vs_sma_5d": "Short-term distance from trend.",
    "close_vs_sma_20d": "One-month distance from trend.",
    "close_vs_sma_50d": "Intermediate trend distance.",
    "close_vs_sma_200d": "Long-term trend distance.",
    "body_pct": "Direction and size of the daily candle body.",
    "volatility_30d": "Realized volatility over the last month.",
    "volatility_60d": "Realized volatility over the last two months.",
    "volatility_5d": "Very recent realized volatility.",
    "volume_ratio_30d": "Liquidity/participation versus recent average.",
    "rsi_14": "Momentum oscillator used as a non-linear context feature.",
}


@dataclass
class TreeNode:
    proba: float
    n_samples: int
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class DecisionTreeClassifier:
    def __init__(
        self,
        *,
        max_depth: int,
        min_samples_leaf: int,
        max_features: int,
        n_thresholds: int,
        rng: np.random.Generator,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.rng = rng
        self.root: TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root = self._build(x, y, depth=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree must be fit before prediction.")
        return np.array([self._predict_one(row, self.root) for row in x])

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> float:
        while node.feature is not None and node.threshold is not None:
            if row[node.feature] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return node.proba

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode(proba=float(np.mean(y)), n_samples=int(y.size))
        if (
            depth >= self.max_depth
            or y.size < self.min_samples_leaf * 2
            or np.all(y == y[0])
        ):
            return node

        split = self._best_split(x, y)
        if split is None:
            return node

        feature, threshold, gain = split
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return node

        if self.feature_importances_ is not None:
            self.feature_importances_[feature] += gain * y.size

        node.feature = feature
        node.threshold = float(threshold)
        node.left = self._build(x[left_mask], y[left_mask], depth + 1)
        node.right = self._build(x[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float, float] | None:
        parent_gini = gini(y)
        best_feature = -1
        best_threshold = 0.0
        best_gain = 0.0
        feature_count = x.shape[1]
        feature_idx = self.rng.choice(
            feature_count, size=min(self.max_features, feature_count), replace=False
        )

        for feature in feature_idx:
            values = x[:, feature]
            thresholds = candidate_thresholds(values, self.n_thresholds)
            for threshold in thresholds:
                left = values <= threshold
                left_count = int(left.sum())
                right_count = y.size - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                weighted_gini = (
                    left_count / y.size * gini(y[left])
                    + right_count / y.size * gini(y[~left])
                )
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_gain = float(gain)

        if best_feature < 0 or best_gain <= 1e-12:
            return None
        return best_feature, best_threshold, best_gain


class RandomForestClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 120,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        max_features: int | None = None,
        n_thresholds: int = 12,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.trees: list[DecisionTreeClassifier] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        rng = np.random.default_rng(self.random_state)
        max_features = self.max_features or max(1, int(math.sqrt(x.shape[1])))
        self.trees = []
        importances = np.zeros(x.shape[1], dtype=float)

        for _ in range(self.n_estimators):
            sample_idx = rng.integers(0, x.shape[0], size=x.shape[0])
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                n_thresholds=self.n_thresholds,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
            )
            tree.fit(x[sample_idx], y[sample_idx])
            self.trees.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest must be fit before prediction.")
        votes = np.vstack([tree.predict_proba(x) for tree in self.trees])
        return votes.mean(axis=0)


def gini(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    p = float(np.mean(y))
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def candidate_thresholds(values: np.ndarray, n_thresholds: int) -> np.ndarray:
    unique = np.unique(values[np.isfinite(values)])
    if unique.size <= 1:
        return np.array([], dtype=float)
    if unique.size <= n_thresholds:
        return (unique[:-1] + unique[1:]) / 2.0
    quantiles = np.linspace(0.05, 0.95, n_thresholds)
    return np.unique(np.quantile(unique, quantiles))


def load_history(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="Historical Data")
    column_order = ["Date", "Open", "High", "Low", "Close", "Volume"]
    expected = set(column_order)
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Historical Data sheet is missing columns: {sorted(missing)}")

    df = df[column_order].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    open_ = out["Open"]
    volume = out["Volume"].replace(0, np.nan)

    out["log_return_1d"] = np.log(close).diff()
    out["next_return"] = close.shift(-1) / close - 1.0
    out["target_up"] = (out["next_return"] > 0).astype(float)
    out.loc[out["next_return"].isna(), "target_up"] = np.nan

    for window in [2, 3, 5, 7, 10, 14, 21, 30, 60, 90]:
        out[f"return_{window}d"] = close / close.shift(window) - 1.0

    for window in [5, 10, 20, 30, 60]:
        out[f"volatility_{window}d"] = out["log_return_1d"].rolling(window).std()
        out[f"volume_ratio_{window}d"] = volume / volume.rolling(window).mean()

    for window in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(window).mean()
        out[f"close_vs_sma_{window}d"] = close / sma - 1.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_pct"] = macd / close
    out["macd_signal_pct"] = macd_signal / close
    out["macd_hist_pct"] = (macd - macd_signal) / close

    out["rsi_14"] = rsi(close, 14)
    out["range_pct"] = (high - low) / close
    out["body_pct"] = (close - open_) / open_
    out["close_position"] = (close - low) / (high - low).replace(0, np.nan)
    out["atr_14_pct"] = atr(out, 14) / close
    out["drawdown_30d"] = close / close.rolling(30).max() - 1.0
    out["drawdown_90d"] = close / close.rolling(90).max() - 1.0
    out["day_of_week"] = out["Date"].dt.dayofweek

    feature_cols = [
        col
        for col in out.columns
        if col
        not in {
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "next_return",
            "target_up",
        }
    ]
    feature_frame = out.replace([np.inf, -np.inf], np.nan)
    model_frame = feature_frame.dropna(
        subset=feature_cols + ["next_return", "target_up"]
    ).reset_index(drop=True)
    live_frame = feature_frame.dropna(subset=feature_cols).reset_index(drop=True)
    return model_frame, live_frame, feature_cols


def rsi(close: pd.Series, window: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window).mean()
    loss = (-diff.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def standardize_train_apply(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def walk_forward_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    initial_train_days: int,
    retrain_every_days: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    importances = np.zeros(len(feature_cols), dtype=float)
    importance_count = 0

    feature_matrix = df[feature_cols].to_numpy(dtype=float)
    target = df["target_up"].to_numpy(dtype=int)
    n = len(df)

    for train_end in range(initial_train_days, n, retrain_every_days):
        test_end = min(train_end + retrain_every_days, n)
        x_train_raw = feature_matrix[:train_end]
        y_train = target[:train_end]
        x_test_raw = feature_matrix[train_end:test_end]
        if x_test_raw.size == 0:
            continue

        x_train, x_test = standardize_train_apply(x_train_raw, x_test_raw)
        forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_thresholds=12,
            random_state=random_state + train_end,
        )
        forest.fit(x_train, y_train)
        proba = forest.predict_proba(x_test)
        block = df.iloc[train_end:test_end].copy()
        block["prob_up"] = proba
        block["prediction"] = (proba >= 0.5).astype(int)
        block["train_end_date"] = df.iloc[train_end - 1]["Date"]
        rows.append(block)

        if forest.feature_importances_ is not None and forest.feature_importances_.sum() > 0:
            importances += forest.feature_importances_
            importance_count += 1

    if not rows:
        raise ValueError("Not enough rows for the requested initial training window.")

    predictions = pd.concat(rows, ignore_index=True)
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances / max(1, importance_count),
        }
    ).sort_values("importance", ascending=False)
    return predictions, importance_df


def backtest(
    predictions: pd.DataFrame,
    *,
    enter_threshold: float,
    exit_threshold: float,
    fee_bps: float,
) -> pd.DataFrame:
    out = predictions.copy()
    fee = fee_bps / 10_000.0
    position = []
    current = 0
    for prob in out["prob_up"]:
        if prob >= enter_threshold:
            current = 1
        elif prob <= exit_threshold:
            current = 0
        position.append(current)

    out["position"] = position
    out["turnover"] = out["position"].diff().abs().fillna(out["position"]).astype(float)
    out["strategy_return"] = out["position"] * out["next_return"] - out["turnover"] * fee
    out["buyhold_return"] = out["next_return"]
    out["strategy_equity"] = (1.0 + out["strategy_return"]).cumprod()
    out["buyhold_equity"] = (1.0 + out["buyhold_return"]).cumprod()
    return out


def classification_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["target_up"].astype(int).to_numpy()
    pred = predictions["prediction"].astype(int).to_numpy()
    prob = predictions["prob_up"].to_numpy(dtype=float)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "accuracy": safe_div(tp + tn, len(y)),
        "precision_up": safe_div(tp, tp + fp),
        "recall_up": safe_div(tp, tp + fn),
        "auc": auc_score(y, prob),
        "positive_rate_actual": float(y.mean()),
        "positive_rate_predicted": float(pred.mean()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def trading_metrics(bt: pd.DataFrame, return_col: str, equity_col: str) -> dict[str, float]:
    returns = bt[return_col].to_numpy(dtype=float)
    equity = bt[equity_col].to_numpy(dtype=float)
    days = max(1, len(returns))
    total_return = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (365.0 / days) - 1.0)
    vol = float(np.std(returns, ddof=1) * math.sqrt(365.0)) if len(returns) > 1 else 0.0
    sharpe = safe_div(float(np.mean(returns) * 365.0), vol)
    dd = max_drawdown(equity)
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe_0rf": sharpe,
        "max_drawdown": dd,
        "best_day": float(np.max(returns)),
        "worst_day": float(np.min(returns)),
    }


def trade_summary(bt: pd.DataFrame) -> dict[str, float]:
    exposure = float(bt["position"].mean())
    entries = int(((bt["position"].diff().fillna(bt["position"]) == 1)).sum())
    exits = int(((bt["position"].diff().fillna(0) == -1)).sum())
    return {
        "exposure": exposure,
        "entries": entries,
        "exits": exits,
        "turnover_events": int(bt["turnover"].sum()),
        "active_days": int(bt["position"].sum()),
    }


def threshold_sensitivity(predictions: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    enter_thresholds = [0.50, 0.525, 0.55, 0.575, 0.60]
    exit_thresholds = [0.45, 0.48, 0.50, 0.525]
    for enter in enter_thresholds:
        for exit_ in exit_thresholds:
            if exit_ > enter:
                continue
            bt = backtest(
                predictions,
                enter_threshold=enter,
                exit_threshold=exit_,
                fee_bps=fee_bps,
            )
            strategy = trading_metrics(bt, "strategy_return", "strategy_equity")
            trades = trade_summary(bt)
            rows.append(
                {
                    "enter_threshold": enter,
                    "exit_threshold": exit_,
                    "total_return": strategy["total_return"],
                    "cagr": strategy["cagr"],
                    "sharpe_0rf": strategy["sharpe_0rf"],
                    "max_drawdown": strategy["max_drawdown"],
                    "exposure": trades["exposure"],
                    "entries": trades["entries"],
                    "turnover_events": trades["turnover_events"],
                }
            )
    return pd.DataFrame(rows).sort_values("sharpe_0rf", ascending=False)


def standalone_action(prob_up: float, enter_threshold: float, exit_threshold: float) -> str:
    if prob_up >= enter_threshold:
        return "LONG"
    if prob_up <= exit_threshold:
        return "CASH"
    return "HOLD_EXISTING_POSITION"


def confidence_band(prob_up: float, enter_threshold: float, exit_threshold: float) -> dict[str, float | str]:
    action = standalone_action(prob_up, enter_threshold, exit_threshold)
    if action == "LONG":
        margin = prob_up - enter_threshold
        label = "Strong Long" if margin >= 0.10 else "Weak Long"
    elif action == "CASH":
        margin = exit_threshold - prob_up
        label = "Strong Cash" if margin >= 0.10 else "Weak Cash"
    else:
        midpoint = (enter_threshold + exit_threshold) / 2.0
        margin = -abs(prob_up - midpoint)
        label = "Neutral / Hold Zone"

    return {
        "label": label,
        "action": action,
        "probability": prob_up,
        "meter_pct": prob_up,
        "distance_from_entry": prob_up - enter_threshold,
        "distance_from_exit": prob_up - exit_threshold,
        "margin": margin,
    }


def percentile_rank(series: pd.Series, value: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or not np.isfinite(value):
        return float("nan")
    return float((clean <= value).mean())


def percentile_reading(percentile: float) -> str:
    if not np.isfinite(percentile):
        return "Unknown"
    if percentile >= 0.80:
        return "Elevated"
    if percentile <= 0.20:
        return "Depressed"
    return "Normal"


def latest_feature_context(
    live_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    importances: pd.DataFrame,
    *,
    limit: int = 8,
) -> list[dict[str, float | str]]:
    latest = live_df.iloc[-1]
    rows: list[dict[str, float | str]] = []
    for _, item in importances.head(limit).iterrows():
        feature = str(item["feature"])
        if feature not in live_df.columns:
            continue
        value = float(latest[feature])
        percentile = percentile_rank(reference_df[feature], value)
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "value": value,
                "importance": float(item["importance"]),
                "percentile": percentile,
                "reading": percentile_reading(percentile),
                "note": FEATURE_NOTES.get(feature, "Current feature reading relative to history."),
            }
        )
    return rows


def classify_market_regime(row: pd.Series, reference_df: pd.DataFrame) -> dict[str, float | str]:
    vol = float(row.get("volatility_30d", np.nan))
    vol_pct = percentile_rank(reference_df["volatility_30d"], vol) if "volatility_30d" in reference_df else float("nan")
    volume_ratio = float(row.get("volume_ratio_30d", np.nan))
    close_vs_50 = float(row.get("close_vs_sma_50d", np.nan))
    close_vs_200 = float(row.get("close_vs_sma_200d", np.nan))
    return_30d = float(row.get("return_30d", np.nan))

    tags: list[str] = []
    if np.isfinite(vol_pct):
        if vol_pct >= 0.80:
            tags.append("High Volatility")
        elif vol_pct <= 0.25:
            tags.append("Low Volatility")
    if np.isfinite(volume_ratio) and volume_ratio < 0.70:
        tags.append("Low Liquidity")

    trend_up = close_vs_50 > 0 and close_vs_200 > 0 and return_30d > 0
    trend_down = close_vs_50 < 0 and close_vs_200 < 0 and return_30d < 0
    trend_strength = max(abs(close_vs_50), abs(return_30d)) if np.isfinite(close_vs_50) and np.isfinite(return_30d) else float("nan")

    if trend_up:
        label = "Trending Up"
    elif trend_down:
        label = "Trending Down"
    elif np.isfinite(trend_strength) and trend_strength < 0.08:
        label = "Choppy"
    elif np.isfinite(vol_pct) and vol_pct >= 0.75:
        label = "High Volatility"
    else:
        label = "Mixed"

    if not tags:
        tags.append("Normal Liquidity")

    return {
        "label": label,
        "tags": ", ".join(tags),
        "volatility_percentile": vol_pct,
        "volume_ratio_30d": volume_ratio,
        "close_vs_sma_50d": close_vs_50,
        "close_vs_sma_200d": close_vs_200,
        "return_30d": return_30d,
    }


def apply_regime_columns(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    regimes = [classify_market_regime(row, reference_df) for _, row in out.iterrows()]
    out["market_regime"] = [str(item["label"]) for item in regimes]
    out["regime_tags"] = [str(item["tags"]) for item in regimes]
    out["volatility_percentile"] = [float(item["volatility_percentile"]) for item in regimes]
    return out


def build_daily_signal_archive(
    bt: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    enter_threshold: float,
    exit_threshold: float,
) -> pd.DataFrame:
    archive = apply_regime_columns(bt, reference_df)
    archive["standalone_signal"] = archive["prob_up"].apply(
        lambda value: standalone_action(float(value), enter_threshold, exit_threshold)
    )
    bands = [
        confidence_band(float(prob), enter_threshold, exit_threshold)
        for prob in archive["prob_up"]
    ]
    archive["confidence_label"] = [str(item["label"]) for item in bands]
    archive["next_day_result"] = np.where(archive["next_return"] > 0, "UP", "DOWN")
    archive["model_was_right"] = archive["prediction"].astype(int) == archive["target_up"].astype(int)
    archive["position_state"] = np.where(archive["position"].astype(int) == 1, "LONG", "CASH")
    return archive[
        [
            "Date",
            "Close",
            "prob_up",
            "standalone_signal",
            "confidence_label",
            "position_state",
            "market_regime",
            "regime_tags",
            "next_return",
            "next_day_result",
            "model_was_right",
            "strategy_return",
            "strategy_equity",
            "buyhold_equity",
        ]
    ].copy()


def build_trade_journal(bt: pd.DataFrame, *, fee_bps: float) -> pd.DataFrame:
    position = bt["position"].astype(int)
    changes = position.diff().fillna(position)
    fee = fee_bps / 10_000.0
    rows: list[dict[str, float | int | str | bool]] = []
    entry_idx: int | None = None

    for idx, change in enumerate(changes):
        if int(change) == 1 and entry_idx is None:
            entry_idx = idx
        is_exit = int(change) == -1
        is_final_open = entry_idx is not None and idx == len(bt) - 1 and position.iloc[idx] == 1
        if entry_idx is None or not (is_exit or is_final_open):
            continue

        exit_idx = idx
        trade = bt.iloc[entry_idx : exit_idx + 1].copy()
        entry = bt.iloc[entry_idx]
        exit_ = bt.iloc[exit_idx]
        close_path = trade["Close"].to_numpy(dtype=float)
        running_max = np.maximum.accumulate(close_path)
        trade_drawdown = float(np.min(close_path / running_max - 1.0))
        gross_return = float(exit_["Close"] / entry["Close"] - 1.0)
        net_return = float((1.0 + trade["strategy_return"]).prod() - 1.0)
        holding_days = max(1, int((pd.Timestamp(exit_["Date"]) - pd.Timestamp(entry["Date"])).days))

        rows.append(
            {
                "trade_id": len(rows) + 1,
                "entry_date": str(pd.Timestamp(entry["Date"]).date()),
                "exit_date": "" if is_final_open else str(pd.Timestamp(exit_["Date"]).date()),
                "status": "OPEN" if is_final_open else "CLOSED",
                "holding_days": holding_days,
                "entry_price": float(entry["Close"]),
                "exit_price": float(exit_["Close"]),
                "entry_probability": float(entry["prob_up"]),
                "exit_probability": float(exit_["prob_up"]),
                "gross_return": gross_return,
                "net_return_after_fees": net_return,
                "max_drawdown": trade_drawdown,
                "fees_estimated": fee if is_final_open else fee * 2.0,
                "winning_trade": bool(net_return > 0),
            }
        )

        entry_idx = None if is_exit else entry_idx

    return pd.DataFrame(rows)


def latest_model_signal(
    model_df: pd.DataFrame,
    live_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    enter_threshold: float,
    exit_threshold: float,
    random_state: int,
) -> dict[str, object]:
    latest = live_df.iloc[[-1]].copy()
    x_train_raw = model_df[feature_cols].to_numpy(dtype=float)
    y_train = model_df["target_up"].to_numpy(dtype=int)
    x_live_raw = latest[feature_cols].to_numpy(dtype=float)
    x_train, x_live = standardize_train_apply(x_train_raw, x_live_raw)

    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_thresholds=12,
        random_state=random_state + 999_983,
    )
    forest.fit(x_train, y_train)
    prob_up = float(forest.predict_proba(x_live)[0])
    confidence = confidence_band(prob_up, enter_threshold, exit_threshold)

    return {
        "as_of_date": str(latest.iloc[0]["Date"].date()),
        "close": float(latest.iloc[0]["Close"]),
        "probability_next_day_up": prob_up,
        "standalone_action": confidence["action"],
        "confidence_label": confidence["label"],
        "confidence_margin": confidence["margin"],
        "distance_from_entry": confidence["distance_from_entry"],
        "distance_from_exit": confidence["distance_from_exit"],
        "entry_threshold": enter_threshold,
        "exit_threshold": exit_threshold,
        "note": "Standalone action does not know your current portfolio position.",
    }


def max_drawdown(equity: Iterable[float]) -> float:
    arr = np.asarray(list(equity), dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdown = arr / running_max - 1.0
    return float(drawdown.min())


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = 0.5 * (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + ties)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def maybe_write_chart(bt: pd.DataFrame, output_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    path = output_dir / "kaspa_rf_equity_curve.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bt["Date"], bt["strategy_equity"], label="Random Forest strategy")
    ax.plot(bt["Date"], bt["buyhold_equity"], label="Buy and hold", alpha=0.75)
    ax.set_title("KASPA Walk-Forward Random Forest Backtest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = DEFAULT_CANONICAL_CSV if DEFAULT_CANONICAL_CSV.exists() else DEFAULT_INPUT
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-train-days", type=int, default=365)
    parser.add_argument("--retrain-every-days", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--enter-threshold", type=float, default=0.55)
    parser.add_argument("--exit-threshold", type=float, default=0.50)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_history(args.input)
    data, live_features, feature_cols = add_features(raw)
    predictions, importances = walk_forward_predictions(
        data,
        feature_cols,
        initial_train_days=args.initial_train_days,
        retrain_every_days=args.retrain_every_days,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    bt = backtest(
        predictions,
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
        fee_bps=args.fee_bps,
    )
    live_signal = latest_model_signal(
        data,
        live_features,
        feature_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
        random_state=args.random_state,
    )
    sensitivity = threshold_sensitivity(predictions, fee_bps=args.fee_bps)
    daily_archive = build_daily_signal_archive(
        bt,
        live_features,
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
    )
    trade_journal = build_trade_journal(bt, fee_bps=args.fee_bps)
    latest_regime = classify_market_regime(live_features.iloc[-1], live_features)
    context_features = latest_feature_context(live_features, live_features, importances)

    metrics = {
        "input_file": str(args.input),
        "data_start": str(raw["Date"].min().date()),
        "data_end": str(raw["Date"].max().date()),
        "raw_rows": int(len(raw)),
        "model_rows_after_feature_engineering": int(len(data)),
        "backtest_start": str(bt["Date"].min().date()),
        "backtest_end": str(bt["Date"].max().date()),
        "backtest_days": int(len(bt)),
        "settings": {
            "initial_train_days": args.initial_train_days,
            "retrain_every_days": args.retrain_every_days,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "enter_threshold": args.enter_threshold,
            "exit_threshold": args.exit_threshold,
            "fee_bps": args.fee_bps,
            "long_only": True,
            "shorting": False,
        },
        "classification": classification_metrics(bt),
        "strategy": trading_metrics(bt, "strategy_return", "strategy_equity"),
        "buy_and_hold": trading_metrics(bt, "buyhold_return", "buyhold_equity"),
        "trades": trade_summary(bt),
        "latest_signal": live_signal,
        "latest_market_regime": latest_regime,
        "latest_feature_context": context_features,
        "best_threshold_sensitivity": sensitivity.head(5).to_dict(orient="records"),
        "top_features": importances.head(12).to_dict(orient="records"),
    }

    equity_path = args.output_dir / "kaspa_rf_equity_curve.csv"
    importance_path = args.output_dir / "kaspa_rf_feature_importance.csv"
    sensitivity_path = args.output_dir / "kaspa_rf_threshold_sensitivity.csv"
    trade_journal_path = args.output_dir / "kaspa_rf_trade_journal.csv"
    daily_archive_path = args.output_dir / "kaspa_rf_daily_signal_archive.csv"
    metrics_path = args.output_dir / "kaspa_rf_metrics.json"
    bt.to_csv(equity_path, index=False)
    importances.to_csv(importance_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    trade_journal.to_csv(trade_journal_path, index=False)
    daily_archive.to_csv(daily_archive_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    chart_path = maybe_write_chart(bt, args.output_dir)
    if chart_path:
        metrics["chart"] = chart_path
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
