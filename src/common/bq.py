"""
Shared BigQuery helpers.

Every script in this project goes through here so credentials,
project id, and load behaviour live in exactly one place.

Local auth:  GOOGLE_APPLICATION_CREDENTIALS points at gcp-key.json
Render auth: GCP_SERVICE_ACCOUNT_JSON holds the key contents; this
             module writes it to a temp file on first use.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("BQ_LOCATION", "US")

_client = None


def _ensure_credentials():
    """Materialise credentials from an env var when running on Render."""

    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    inline = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    if not inline:
        raise EnvironmentError(
            "No credentials found. Set GOOGLE_APPLICATION_CREDENTIALS "
            "to a key file path, or GCP_SERVICE_ACCOUNT_JSON to the key contents."
        )

    # fail early on malformed JSON rather than deep inside the client
    json.loads(inline)

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    handle.write(inline)
    handle.close()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = handle.name


def get_client() -> bigquery.Client:
    """Return a cached BigQuery client."""

    global _client

    if _client is None:
        if not PROJECT_ID:
            raise EnvironmentError("GCP_PROJECT_ID is not set.")

        _ensure_credentials()
        _client = bigquery.Client(project=PROJECT_ID, location=LOCATION)

    return _client


def table_id(dataset: str, table: str) -> str:
    """Fully qualified table reference."""

    return f"{PROJECT_ID}.{dataset}.{table}"


def query_df(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a query and return the results as a DataFrame."""

    config = bigquery.QueryJobConfig(query_parameters=params or [])
    return get_client().query(sql, job_config=config).result().to_dataframe()


def load_dataframe(
    df: pd.DataFrame,
    dataset: str,
    table: str,
    write_disposition: str = "WRITE_TRUNCATE",
    schema: list | None = None,
) -> int:
    """
    Load a DataFrame into a table.

    WRITE_TRUNCATE  replace the table contents
    WRITE_APPEND    add rows
    WRITE_EMPTY     fail if the table already has rows
    """

    client = get_client()

    config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        schema=schema,
        autodetect=schema is None,
    )

    job = client.load_table_from_dataframe(
        df, table_id(dataset, table), job_config=config
    )
    job.result()

    return len(df)


def merge_via_staging(
    df: pd.DataFrame,
    dataset: str,
    target: str,
    key_columns: list[str],
    update_columns: list[str] | None = None,
    schema: list | None = None,
) -> None:
    """
    Upsert a DataFrame into a target table.

    Loads into a temporary staging table, MERGEs on key_columns, then
    drops the staging table. Idempotent on re-runs.
    """

    client = get_client()
    staging = f"_stg_{target}"

    load_dataframe(df, dataset, staging, "WRITE_TRUNCATE", schema)

    all_columns = list(df.columns)
    update_columns = update_columns or [
        c for c in all_columns if c not in key_columns
    ]

    on_clause = " AND ".join(f"T.{c} = S.{c}" for c in key_columns)
    set_clause = ", ".join(f"{c} = S.{c}" for c in update_columns)
    insert_cols = ", ".join(all_columns)
    insert_vals = ", ".join(f"S.{c}" for c in all_columns)

    matched = (
        f"WHEN MATCHED THEN UPDATE SET {set_clause}" if update_columns else ""
    )

    sql = f"""
        MERGE `{table_id(dataset, target)}` T
        USING `{table_id(dataset, staging)}` S
        ON {on_clause}
        {matched}
        WHEN NOT MATCHED THEN
            INSERT ({insert_cols}) VALUES ({insert_vals})
    """

    client.query(sql).result()
    client.query(f"DROP TABLE `{table_id(dataset, staging)}`").result()


def ensure_table(dataset: str, table: str, ddl_columns: str) -> None:
    """Create a table if it does not exist. ddl_columns is the column list."""

    sql = f"""
        CREATE TABLE IF NOT EXISTS `{table_id(dataset, table)}` (
            {ddl_columns}
        )
    """
    get_client().query(sql).result()