"""
Ingests BTC-USD, SPY, and DX-Y.NYB daily prices into BigQuery raw.btc_prices.

BTC trades 24/7 while SPY and DXY do not, so the calendar is reindexed to
every day and the equity/dollar series are forward filled. Idempotent:
re-running merges on price_date rather than duplicating.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bq import ensure_table, merge_via_staging, query_df, table_id

DATASET = "raw"
TABLE = "btc_prices"

SCHEMA = [
    bigquery.SchemaField("price_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("btc_high", "FLOAT64"),
    bigquery.SchemaField("btc_low", "FLOAT64"),
    bigquery.SchemaField("btc_close", "FLOAT64"),
    bigquery.SchemaField("btc_volume", "FLOAT64"),
    bigquery.SchemaField("spy_close", "FLOAT64"),
    bigquery.SchemaField("dxy_close", "FLOAT64"),
]

DDL = """
    price_date DATE NOT NULL,
    btc_high   FLOAT64,
    btc_low    FLOAT64,
    btc_close  FLOAT64,
    btc_volume FLOAT64,
    spy_close  FLOAT64,
    dxy_close  FLOAT64
"""


def fetch_prices(days_back: int = 90) -> pd.DataFrame:
    tickers = ["BTC-USD", "SPY", "DX-Y.NYB"]
    raw = yf.download(tickers, period=f"{days_back}d", interval="1d", progress=False)

    rows = []
    for dt in raw.index:
        rows.append(
            {
                "price_date": dt.date(),
                "btc_high": float(raw["High"]["BTC-USD"][dt]),
                "btc_low": float(raw["Low"]["BTC-USD"][dt]),
                "btc_close": float(raw["Close"]["BTC-USD"][dt]),
                "btc_volume": float(raw["Volume"]["BTC-USD"][dt]),
                "spy_close": float(raw["Close"]["SPY"][dt]),
                "dxy_close": float(raw["Close"]["DX-Y.NYB"][dt]),
            }
        )

    df = pd.DataFrame(rows)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df = df.set_index("price_date")

    # BTC trades every day; reindex the calendar and forward fill the gaps
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range).ffill()

    df = df.dropna()
    df.index.name = "price_date"
    df = df.reset_index()
    df["price_date"] = df["price_date"].dt.date

    today_utc = pd.Timestamp.now(tz="UTC").date()
    df = df[df["price_date"] < today_utc].reset_index(drop=True)

    return df


def run(days_back: int = 90):
    df = fetch_prices(days_back)
    print(f"Fetched {len(df)} rows ({df['price_date'].min()} to {df['price_date'].max()})")

    ensure_table(DATASET, TABLE, DDL)

    merge_via_staging(
        df,
        dataset=DATASET,
        target=TABLE,
        key_columns=["price_date"],
        schema=SCHEMA,
    )

    total = query_df(
        f"SELECT COUNT(*) AS n FROM `{table_id(DATASET, TABLE)}`"
    )["n"].iloc[0]

    print(f"Merged. raw.btc_prices now holds {total} rows.")


if __name__ == "__main__":
    run(days_back=4400)