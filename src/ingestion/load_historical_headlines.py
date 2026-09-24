"""
One-time historical load.

Reads all_time_labeled_headlines_with_dates.csv and populates both
raw.crypto_headlines and raw.headline_sentiments.

The labels in this CSV are the ones the LSTM's sentiment feature was
built from, so loading them directly keeps serving consistent with
training. Daily headlines from RSS get labelled by the HF Space instead.
"""

import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bq import ensure_table, merge_via_staging, query_df, table_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "all_time_labeled_headlines_with_dates.csv"

DATASET = "raw"

HEADLINES_DDL = """
    link          STRING NOT NULL,
    headline      STRING,
    source        STRING,
    published_raw STRING
"""

SENTIMENTS_DDL = """
    link            STRING NOT NULL,
    sentiment_score INT64
"""

HEADLINES_SCHEMA = [
    bigquery.SchemaField("link", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("headline", "STRING"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("published_raw", "STRING"),
]

SENTIMENTS_SCHEMA = [
    bigquery.SchemaField("link", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("sentiment_score", "INT64"),
]

LABEL_MAP = {
    "bearish": -1, "negative": -1,
    "neutral": 0,
    "bullish": 1, "positive": 1,
}


def run():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower()
    print(f"Read {len(df)} rows from {CSV_PATH.name}")
    print(f"Columns: {list(df.columns)}")

    headline_col = "headlines" if "headlines" in df.columns else "headline"

    # synthetic key: the CSV has no URLs
    df["link"] = [f"csv://{i}" for i in range(len(df))]

    headlines = pd.DataFrame(
        {
            "link": df["link"],
            "headline": df[headline_col].astype(str),
            "source": "historical_csv",
            "published_raw": df["date"].astype(str),
        }
    )

    sentiments = pd.DataFrame(
        {
            "link": df["link"],
            "sentiment_score": (
                df["sentiment"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(LABEL_MAP)
            ),
        }
    )

    unmapped = sentiments["sentiment_score"].isna().sum()
    if unmapped:
        bad = df.loc[sentiments["sentiment_score"].isna(), "sentiment"].unique()[:5]
        raise ValueError(
            f"{unmapped} rows have a sentiment label that is not in LABEL_MAP. "
            f"Examples: {list(bad)}"
        )

    sentiments["sentiment_score"] = sentiments["sentiment_score"].astype("int64")

    ensure_table(DATASET, "crypto_headlines", HEADLINES_DDL)
    ensure_table(DATASET, "headline_sentiments", SENTIMENTS_DDL)

    merge_via_staging(
        headlines,
        dataset=DATASET,
        target="crypto_headlines",
        key_columns=["link"],
        schema=HEADLINES_SCHEMA,
    )
    print(f"Merged {len(headlines)} headlines.")

    merge_via_staging(
        sentiments,
        dataset=DATASET,
        target="headline_sentiments",
        key_columns=["link"],
        schema=SENTIMENTS_SCHEMA,
    )
    print(f"Merged {len(sentiments)} sentiment labels.")

    counts = query_df(
        f"""
        SELECT
            (SELECT COUNT(*) FROM `{table_id(DATASET, 'crypto_headlines')}`)    AS headlines,
            (SELECT COUNT(*) FROM `{table_id(DATASET, 'headline_sentiments')}`) AS sentiments
        """
    )
    print(counts.to_string(index=False))

    dist = query_df(
        f"""
        SELECT sentiment_score, COUNT(*) AS n
        FROM `{table_id(DATASET, 'headline_sentiments')}`
        GROUP BY sentiment_score
        ORDER BY sentiment_score
        """
    )
    print("\nLabel distribution:")
    print(dist.to_string(index=False))


if __name__ == "__main__":
    run()