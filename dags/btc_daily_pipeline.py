"""
Daily BTC direction pipeline.

    ingest_prices ─────────────────────┐
    ingest_news ──> label_sentiment ───┴─> dbt_staging ─> build_features ─> predict ─> dbt_performance

Every task shells out to the pipeline's own virtualenv. Airflow only
orchestrates and never imports pipeline code, so its dependencies can't
collide with dbt, torch, or the BigQuery client.

Production runs on GitHub Actions via `airflow dags test` (see
.github/workflows/daily-pipeline.yml). The same file runs unchanged under
a normal Airflow scheduler on the same cron.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

ROOT = os.environ.get("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
PY = os.environ.get("PIPELINE_PY", "python")
DBT = os.environ.get("DBT_BIN", "dbt")


def run_script(path: str) -> str:
    return f'"{PY}" {path}'


def run_dbt(select: str) -> str:
    return f'"{DBT}" run --select {select} --profiles-dir .'


default_args = {
    "owner": "hiyaw",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="btc_daily_pipeline",
    description="Ingest BTC prices and news, label sentiment, build features, predict 3-day direction",
    # 01:30 UTC: the daily BTC candle closes at 00:00 UTC
    schedule="30 1 * * *",
    start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["btc", "ml", "pipeline"],
    doc_md=__doc__,
) as dag:

    ingest_prices = BashOperator(
        task_id="ingest_prices",
        bash_command=run_script("src/ingestion/ingest_prices.py"),
        cwd=ROOT,
    )

    ingest_news = BashOperator(
        task_id="ingest_news",
        bash_command=run_script("src/ingestion/ingest_news.py"),
        cwd=ROOT,
    )

    label_sentiment = BashOperator(
        task_id="label_sentiment",
        bash_command=run_script("src/ingestion/label_sentiment.py"),
        cwd=ROOT,
        execution_timeout=timedelta(minutes=20),
    )

    dbt_staging = BashOperator(
        task_id="dbt_staging",
        bash_command=run_dbt("staging"),
        cwd=f"{ROOT}/dbt",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=run_script("src/ingestion/build_features.py"),
        cwd=ROOT,
    )

    predict = BashOperator(
        task_id="predict",
        bash_command=run_script("src/inference/predict.py"),
        cwd=ROOT,
    )

    dbt_performance = BashOperator(
        task_id="dbt_performance",
        bash_command=run_dbt("mart_prediction_performance"),
        cwd=f"{ROOT}/dbt",
    )

    ingest_news >> label_sentiment
    [ingest_prices, label_sentiment] >> dbt_staging
    dbt_staging >> build_features >> predict >> dbt_performance