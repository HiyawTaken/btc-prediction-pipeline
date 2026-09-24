"""
Pulls crypto headlines from Google News RSS into raw.crypto_headlines.

Runs daily. Chunks the date range because Google caps results per query,
and merges on link so re-runs never duplicate.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import feedparser
import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bq import ensure_table, merge_via_staging, query_df, table_id

DATASET = "raw"
TABLE = "crypto_headlines"

QUERY = "cryptocurrency OR bitcoin OR ethereum OR crypto"
CHUNK_DAYS = 3
SLEEP_BETWEEN = 1.5

DDL = """
    link          STRING NOT NULL,
    headline      STRING,
    source        STRING,
    published_raw STRING
"""

SCHEMA = [
    bigquery.SchemaField("link", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("headline", "STRING"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("published_raw", "STRING"),
]


def fetch_headlines(days_back: int = 7) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=days_back)

    seen, rows = set(), []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end)

        q = (
            f'{QUERY} after:{current.strftime("%Y-%m-%d")} '
            f'before:{chunk_end.strftime("%Y-%m-%d")}'
        )
        url = (
            "https://news.google.com/rss/search?"
            f"q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
        )

        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if entry.link in seen:
                    continue
                seen.add(entry.link)
                rows.append(
                    {
                        "link": entry.link,
                        "headline": entry.title,
                        "source": (
                            getattr(entry.source, "title", "")
                            if hasattr(entry, "source")
                            else ""
                        ),
                        "published_raw": getattr(entry, "published", ""),
                    }
                )
        except Exception as exc:
            print(f"RSS error on {current.date()}: {exc}")

        current = chunk_end
        time.sleep(SLEEP_BETWEEN)

    return pd.DataFrame(rows)


def run(days_back: int = 7):
    print(f"Fetching headlines for the last {days_back} days...")
    df = fetch_headlines(days_back)

    if df.empty:
        print("No headlines returned. Nothing to merge.")
        return

    print(f"Got {len(df)} unique headlines.")

    ensure_table(DATASET, TABLE, DDL)

    merge_via_staging(
        df,
        dataset=DATASET,
        target=TABLE,
        key_columns=["link"],
        schema=SCHEMA,
    )

    total = query_df(
        f"SELECT COUNT(*) AS n FROM `{table_id(DATASET, TABLE)}`"
    )["n"].iloc[0]

    print(f"Merged. raw.crypto_headlines now holds {total} rows.")


if __name__ == "__main__":
    run()