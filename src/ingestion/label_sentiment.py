"""
Labels any headline in raw.crypto_headlines that has no row in
raw.headline_sentiments yet.

Defaults to local inference because ZeroGPU Spaces have a small daily
quota and a backfill of a few thousand headlines blows through it.
Pass --space to route through the deployed Space instead, which is the
right choice for the daily trickle of new headlines.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bq import ensure_table, merge_via_staging, query_df, table_id

DATASET = "raw"
TABLE = "headline_sentiments"

HF_MODEL = "HiyawErtiro/cryptobert-finetuned_on_news_headlines"
HF_SPACE = "HiyawErtiro/cryptobert-api"

DDL = """
    link            STRING NOT NULL,
    sentiment_score INT64
"""

SCHEMA = [
    bigquery.SchemaField("link", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("sentiment_score", "INT64"),
]

LABEL_MAP = {
    "bearish": -1, "negative": -1, "label_0": -1,
    "neutral": 0, "label_1": 0,
    "bullish": 1, "positive": 1, "label_2": 1,
}


def fetch_unlabeled() -> pd.DataFrame:
    return query_df(
        f"""
        SELECT h.link, h.headline
        FROM `{table_id(DATASET, 'crypto_headlines')}` h
        LEFT JOIN `{table_id(DATASET, TABLE)}` s
            ON h.link = s.link
        WHERE s.link IS NULL
          AND h.headline IS NOT NULL
          AND LENGTH(TRIM(h.headline)) > 0
        """
    )


def label_local(headlines: list[str]) -> list[int]:
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading model locally on {'GPU' if device == 0 else 'CPU'}...")

    classifier = pipeline(
        "text-classification",
        model=HF_MODEL,
        device=device,
        top_k=1,
    )

    scores = []
    batch = 64

    for i in range(0, len(headlines), batch):
        chunk = headlines[i : i + batch]
        results = classifier(chunk, truncation=True, max_length=128, batch_size=batch)

        for result in results:
            top = result[0] if isinstance(result, list) else result
            scores.append(LABEL_MAP.get(str(top["label"]).lower(), 0))

        done = min(i + batch, len(headlines))
        if done % 3200 == 0 or done == len(headlines):
            print(f"  labelled {done}/{len(headlines)}")

    return scores


def label_space(headlines: list[str]) -> list[int]:
    from gradio_client import Client

    client = Client(HF_SPACE)
    scores = []
    batch = 16

    for i in range(0, len(headlines), batch):
        chunk = headlines[i : i + batch]
        results = client.predict("\n".join(chunk), api_name="/predict")

        for result in results:
            scores.append(LABEL_MAP.get(str(result["label"]).lower(), 0))

        print(f"  labelled {min(i + batch, len(headlines))}/{len(headlines)}")

    return scores


def run(use_space: bool = False):
    ensure_table(DATASET, TABLE, DDL)

    df = fetch_unlabeled()
    print(f"Found {len(df)} unlabelled headlines.")

    if df.empty:
        print("Nothing to do.")
        return

    headlines = df["headline"].astype(str).tolist()
    scores = label_space(headlines) if use_space else label_local(headlines)

    if len(scores) != len(headlines):
        raise ValueError(
            f"Got {len(scores)} scores for {len(headlines)} headlines. "
            "Refusing to write misaligned labels."
        )

    out = pd.DataFrame(
        {"link": df["link"].values, "sentiment_score": pd.Series(scores, dtype="int64")}
    )

    merge_via_staging(
        out,
        dataset=DATASET,
        target=TABLE,
        key_columns=["link"],
        schema=SCHEMA,
    )

    print(f"\nMerged {len(out)} labels.")

    dist = query_df(
        f"""
        SELECT sentiment_score, COUNT(*) AS n
        FROM `{table_id(DATASET, TABLE)}`
        GROUP BY sentiment_score
        ORDER BY sentiment_score
        """
    )
    print(dist.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--space",
        action="store_true",
        help="route through the HF Space instead of local inference",
    )
    args = parser.parse_args()

    run(use_space=args.space)