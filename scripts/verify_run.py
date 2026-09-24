"""
Fails the workflow unless this run wrote a fresh prediction and refreshed
the performance mart.

`airflow dags test` already exits 1 when a task fails (tested on 3.3.2).
This adds a check on the output itself: that the run actually wrote a
prediction during this run, the prediction is recent, and the dashboard's
mart caught up to it.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.bq import query_df, table_id

MAX_STALENESS_DAYS = 2


def to_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def main() -> int:
    started = to_utc(os.environ["RUN_STARTED_AT"])
    today = datetime.now(timezone.utc).date()

    newest = query_df(
        f"""
        SELECT prediction_date, direction, raw_score, created_at
        FROM `{table_id('marts', 'predictions')}`
        ORDER BY created_at DESC
        LIMIT 1
        """
    )

    perf = query_df(
        f"""
        SELECT MAX(feature_date) AS latest
        FROM `{table_id('marts', 'mart_prediction_performance')}`
        """
    )

    problems = []

    if newest.empty:
        problems.append("marts.predictions is empty")
    else:
        row = newest.iloc[0]
        created = to_utc(row["created_at"])
        pred_date = pd.Timestamp(row["prediction_date"]).date()

        if created < started:
            problems.append(
                f"no prediction was written during this run "
                f"(newest created_at {created}, run started {started})"
            )

        age = (today - pred_date).days
        if age > MAX_STALENESS_DAYS:
            problems.append(f"newest prediction is {age} days old ({pred_date})")

        latest_perf = perf["latest"].iloc[0]
        if latest_perf is None or pd.isna(latest_perf) or pd.Timestamp(latest_perf).date() < pred_date:
            problems.append(
                f"performance mart not refreshed (latest {latest_perf}, prediction {pred_date})"
            )

        print(
            f"Newest prediction: {pred_date}  {row['direction']}  "
            f"raw {row['raw_score']}  written {created}"
        )

    if problems:
        print("\nRun verification FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Run verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())