"""
Backfills historical predictions into marts.predictions.

Only runs over dates after the training split so the stored history is
genuinely out-of-sample. Re-running replaces its own range and leaves
live daily predictions alone.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from common.bq import ensure_table, get_client, load_dataframe, query_df, table_id
from predict import DDL, SCHEMA, SEQ_LEN, load_model, load_scaler

# First date not used for training (the 85% split boundary).
BACKTEST_START = "2024-11-03"

DATASET = "marts"
TABLE = "predictions"


def run():
    feature_cols, mean, std = load_scaler()
    model = load_model()

    columns = ", ".join(["price_date"] + feature_cols)

    df = query_df(
        f"""
        SELECT {columns}
        FROM `{table_id('marts', 'mart_btc_features')}`
        ORDER BY price_date
        """
    )

    print(f"Loaded {len(df)} feature rows.")

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    scaled = (features - mean) / std

    cutoff = pd.Timestamp(BACKTEST_START).date()
    now = datetime.now(timezone.utc)

    rows = []
    skipped = 0

    for end in range(SEQ_LEN - 1, len(df)):
        feature_date = df["price_date"].iloc[end]
        feature_date = (
            feature_date.date() if hasattr(feature_date, "date") else feature_date
        )

        if feature_date < cutoff:
            continue

        window = scaled[end - SEQ_LEN + 1 : end + 1]

        if np.isnan(window).any():
            skipped += 1
            continue

        probability, direction = model.predict(window[np.newaxis, :, :])
        probability = float(probability)
        direction = int(direction)

        rows.append(
            {
                "prediction_date": feature_date,
                "feature_date": feature_date,
                "direction": "BULLISH" if direction == 1 else "BEARISH",
                "confidence": round(
                    probability if direction == 1 else 1.0 - probability, 4
                ),
                "raw_score": round(probability, 4),
                "created_at": now,
            }
        )

    if not rows:
        print("No rows to write. Check BACKTEST_START against the feature date range.")
        return

    out = pd.DataFrame(rows)

    ensure_table(DATASET, TABLE, DDL)

    # clear only this backfill's own range, never live predictions
    get_client().query(
        f"""
        DELETE FROM `{table_id(DATASET, TABLE)}`
        WHERE prediction_date >= DATE('{BACKTEST_START}')
        """
    ).result()

    load_dataframe(
        out,
        dataset=DATASET,
        table=TABLE,
        write_disposition="WRITE_APPEND",
        schema=SCHEMA,
    )

    scores = out["raw_score"].to_numpy()
    bullish = int((out["direction"] == "BULLISH").sum())

    print(
        f"Saved {len(out)} predictions from "
        f"{out['prediction_date'].min()} to {out['prediction_date'].max()}."
    )
    print(f"Skipped {skipped} windows containing nulls.")
    print(f"BULLISH: {bullish}  BEARISH: {len(out) - bullish}")
    print(
        f"Raw score  min {scores.min():.4f}  "
        f"max {scores.max():.4f}  mean {scores.mean():.4f}"
    )


if __name__ == "__main__":
    run()