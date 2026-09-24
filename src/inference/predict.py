"""
Reads marts.mart_btc_features, runs LSTM inference, writes one row
to marts.predictions.
"""

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
MODEL_DIR = PROJECT_ROOT / "models"

SEQ_LEN = 30
MAX_STALENESS_DAYS = 2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from common.bq import ensure_table, merge_via_staging, query_df, table_id
from lstm_numpy import LSTMInference

DATASET = "marts"
TABLE = "predictions"

DDL = """
    prediction_date DATE NOT NULL,
    feature_date    DATE NOT NULL,
    direction       STRING NOT NULL,
    confidence      FLOAT64 NOT NULL,
    raw_score       FLOAT64 NOT NULL,
    created_at      TIMESTAMP
"""

SCHEMA = [
    bigquery.SchemaField("prediction_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("feature_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("direction", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("confidence", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("raw_score", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("created_at", "TIMESTAMP"),
]


def load_scaler():
    """Load feature order and training scaling statistics."""

    scaler_path = MODEL_DIR / "scaler.json"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with scaler_path.open("r", encoding="utf-8") as handle:
        scaler = json.load(handle)

    missing = {"feature_cols", "mean", "std"} - scaler.keys()
    if missing:
        raise ValueError(f"scaler.json is missing keys: {sorted(missing)}")

    feature_cols = [str(c).lower() for c in scaler["feature_cols"]]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)

    if not (len(feature_cols) == len(mean) == len(std)):
        raise ValueError(
            "scaler.json mismatch: feature_cols, mean and std differ in length."
        )

    return feature_cols, mean, np.where(std == 0, 1.0, std)


def load_model():
    model_path = MODEL_DIR / "lstm_best_30days_horizon.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    return LSTMInference(str(model_path))


def load_features(feature_cols: list[str]) -> pd.DataFrame:
    columns = ", ".join(["price_date"] + feature_cols)

    df = query_df(
        f"""
        SELECT {columns}
        FROM `{table_id('marts', 'mart_btc_features')}`
        ORDER BY price_date DESC
        LIMIT {SEQ_LEN}
        """
    )

    if df.empty:
        raise ValueError("mart_btc_features returned no rows.")

    return df.sort_values("price_date").reset_index(drop=True)


def validate(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"mart_btc_features is missing features: {missing}")

    if len(df) < SEQ_LEN:
        raise ValueError(f"Need {SEQ_LEN} rows, got {len(df)}.")

    latest = df["price_date"].iloc[-1]
    latest = latest.date() if hasattr(latest, "date") else latest

    age = (date.today() - latest).days
    if age > MAX_STALENESS_DAYS:
        raise ValueError(
            f"Features are {age} days stale (latest: {latest}). "
            "Ingestion or the feature build likely failed upstream."
        )

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    null_cols = features.columns[features.isna().any()].tolist()
    if null_cols:
        raise ValueError(f"Null or non-numeric values in: {null_cols}")

    return features.to_numpy(dtype=np.float32)


def run():
    feature_cols, mean, std = load_scaler()
    model = load_model()

    df = load_features(feature_cols)
    raw = validate(df, feature_cols)

    sequence = ((raw - mean) / std)[-SEQ_LEN:][np.newaxis, :, :]

    probability, direction = model.predict(sequence)
    probability = float(np.asarray(probability).squeeze())
    direction = int(np.asarray(direction).squeeze())

    if not 0 <= probability <= 1:
        raise ValueError(f"Model returned an invalid probability: {probability}")

    feature_date = df["price_date"].iloc[-1]
    feature_date = feature_date.date() if hasattr(feature_date, "date") else feature_date

    row = pd.DataFrame(
        [
            {
                "prediction_date": feature_date,
                "feature_date": feature_date,
                "direction": "BULLISH" if direction == 1 else "BEARISH",
                "confidence": round(
                    probability if direction == 1 else 1.0 - probability, 4
                ),
                "raw_score": round(probability, 4),
                "created_at": datetime.now(timezone.utc),
            }
        ]
    )

    ensure_table(DATASET, TABLE, DDL)

    merge_via_staging(
        row,
        dataset=DATASET,
        target=TABLE,
        key_columns=["prediction_date"],
        schema=SCHEMA,
    )

    print("Prediction saved")
    print(f"  Feature date: {feature_date}")
    print(f"  Direction:    {row['direction'].iloc[0]}")
    print(f"  Confidence:   {row['confidence'].iloc[0]:.2%}")
    print(f"  Raw score:    {row['raw_score'].iloc[0]}")


if __name__ == "__main__":
    run()