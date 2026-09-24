"""
Builds marts.mart_btc_features.

This was a dbt Python model on Snowflake. BigQuery runs dbt Python models
on Dataproc Serverless, which is not free, so it lives here instead and
runs as its own DAG task between the staging models and prediction.

The feature logic is lifted directly from the training script. RSI, MACD,
ATR and EMA are recursive, so they cannot be expressed as SQL window
functions without drifting from what the model was trained on.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bq import load_dataframe, query_df, table_id

DATASET = "marts"
TABLE = "mart_btc_features"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = query_df(
        f"""
        SELECT price_date, btc_high, btc_low, btc_close, btc_volume,
               spy_close, dxy_close
        FROM `{table_id('staging', 'stg_btc_prices')}`
        ORDER BY price_date
        """
    )

    sentiment = query_df(
        f"""
        SELECT sentiment_date, sentiment_score
        FROM `{table_id('staging', 'stg_daily_sentiment')}`
        ORDER BY sentiment_date
        """
    )

    return prices, sentiment


def build_features(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values("price_date").reset_index(drop=True)
    df["price_date"] = pd.to_datetime(df["price_date"])

    close = df["btc_close"]
    high = df["btc_high"]
    low = df["btc_low"]
    volume = df["btc_volume"]

    # moving averages
    df["sma_10"] = close.rolling(10).mean() / close
    df["sma_50"] = close.rolling(50).mean() / close
    df["ema_20"] = close.ewm(span=20, adjust=False).mean() / close
    df["max_20"] = close.rolling(20).max() / close
    df["min_20"] = close.rolling(20).min() / close

    # anchors
    df["spy_log_return"] = np.log(df["spy_close"] / df["spy_close"].shift(1))
    df["dxy_log_return"] = np.log(df["dxy_close"] / df["dxy_close"].shift(1))

    # lagged returns
    for lag in (1, 3, 7):
        df[f"return_{lag}d"] = close.pct_change(periods=lag)

    # RSI, Wilder smoothing
    delta = close - close.shift(1)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    df["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss))

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = (ema_12 - ema_26) / close
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # ATR from true range
    prev_close = close.shift(1)
    true_range = np.maximum.reduce(
        [
            (high - low).values,
            (high - prev_close).abs().values,
            (low - prev_close).abs().values,
        ]
    )
    df["atr_14"] = (
        pd.Series(true_range, index=df.index).ewm(com=13, adjust=False).mean() / close
    )

    # volume
    df["rel_volume_20"] = volume / volume.rolling(20).mean()
    typical_price = (high + low + close) / 3
    df["rolling_vwap"] = (
        (volume * typical_price).rolling(14).sum() / volume.rolling(14).sum()
    ) / close

    df["log_return"] = np.log(close / close.shift(1))

    # Bollinger
    sma_20_price = close.rolling(20).mean()
    std_20_price = close.rolling(20).std()
    df["sma_20"] = sma_20_price / close
    df["std_20"] = std_20_price / close
    upper = sma_20_price + 2 * std_20_price
    lower = sma_20_price - 2 * std_20_price
    df["bb_distance"] = (close - lower) / (upper - lower)

    # day of week
    dow = df["price_date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # sentiment, forward filled onto price dates
    sentiment = sentiment.copy()
    sentiment["sentiment_date"] = pd.to_datetime(sentiment["sentiment_date"])

    df = df.merge(
        sentiment[["sentiment_date", "sentiment_score"]],
        left_on="price_date",
        right_on="sentiment_date",
        how="left",
    ).drop(columns=["sentiment_date"])

    df["sentiment_score"] = df["sentiment_score"].ffill().bfill()
    df["sentiment_score_lag_1"] = df["sentiment_score"].shift(1)
    df["sentiment_score_rolling_7"] = df["sentiment_score"].rolling(7).mean()
    df["sentiment_score_rolling_30"] = df["sentiment_score"].rolling(30).mean()

    # 3-day forward label, null for rows with no future price yet
    future_return = (close.shift(-3) - close) / close * 100
    df["future_3d_label"] = np.where(
        future_return.isna(), np.nan, np.where(future_return > 0, 1, 0)
    )

    feature_cols = [
        "sma_10", "sma_50", "ema_20", "max_20", "min_20",
        "spy_log_return", "dxy_log_return",
        "return_1d", "return_3d", "return_7d",
        "rsi_14", "macd", "macd_signal", "macd_histogram",
        "atr_14", "rel_volume_20", "rolling_vwap", "log_return",
        "sma_20", "std_20", "bb_distance",
        "dow_sin", "dow_cos",
        "sentiment_score", "sentiment_score_lag_1",
        "sentiment_score_rolling_7", "sentiment_score_rolling_30",
    ]

    # drop warmup rows only; keep recent rows whose label is not resolvable yet
    df = df.dropna(subset=feature_cols)

    out = df[["price_date"] + feature_cols + ["future_3d_label"]].copy()
    out["price_date"] = out["price_date"].dt.date

    return out


def run():
    prices, sentiment = load_inputs()
    print(f"Loaded {len(prices)} price rows, {len(sentiment)} sentiment days.")

    features = build_features(prices, sentiment)

    if features.empty:
        raise ValueError(
            "Feature build produced no rows. Check that staging models are populated."
        )

    schema = [bigquery.SchemaField("price_date", "DATE", mode="REQUIRED")]
    for column in features.columns:
        if column != "price_date":
            schema.append(bigquery.SchemaField(column, "FLOAT64"))

    load_dataframe(
        features,
        dataset=DATASET,
        table=TABLE,
        write_disposition="WRITE_TRUNCATE",
        schema=schema,
    )

    print(
        f"Wrote {len(features)} rows to marts.{TABLE} "
        f"({features['price_date'].min()} to {features['price_date'].max()})"
    )
    print(f"Feature columns: {len(features.columns) - 2}")


if __name__ == "__main__":
    run()