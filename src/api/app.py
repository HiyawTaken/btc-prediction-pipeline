"""
FastAPI dashboard for the BTC direction pipeline.

Reads marts.mart_prediction_performance from BigQuery and serves one page.
Results are cached in memory, so traffic can never cost more than one
BigQuery query per CACHE_TTL_SECONDS. If BigQuery is unreachable, the
last good result keeps being served instead of a 500.
"""

import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# locally this file lives at src/api/app.py, so src/ holds common/
# in the container common/ sits next to this file and is found via cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.bq import query_df, table_id

app = FastAPI(title="BTC 3-Day Direction")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

CACHE_TTL_SECONDS = 600
_cache = {"data": None, "fetched_at": 0.0}


def query_performance() -> list[dict]:
    df = query_df(
        f"""
        SELECT
            feature_date,
            direction,
            confidence,
            raw_score,
            actual_direction,
            is_correct,
            baseline_correct,
            actual_return_pct,
            hit_rate_30d,
            baseline_rate_30d
        FROM `{table_id('marts', 'mart_prediction_performance')}`
        ORDER BY feature_date
        """
    )

    # pandas NA does not survive templating; turn every missing value into None
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict("records")


def build_view_model(rows: list[dict]) -> dict:
    if not rows:
        return {"empty": True}

    latest = rows[-1]

    resolved = [r for r in rows if r["is_correct"] is not None]
    n = len(resolved)

    hits = sum(int(r["is_correct"]) for r in resolved)
    base_hits = sum(int(r["baseline_correct"]) for r in resolved)

    hit_rate = hits / n if n else 0.0
    baseline_rate = base_hits / n if n else 0.0

    # 95% confidence half-width on the hit rate
    margin = 1.96 * math.sqrt(hit_rate * (1 - hit_rate) / n) if n else 0.0

    chart = [
        {
            "date": str(r["feature_date"]),
            "model": float(r["hit_rate_30d"]),
            "baseline": float(r["baseline_rate_30d"]),
        }
        for r in rows
        if r["hit_rate_30d"] is not None and r["baseline_rate_30d"] is not None
    ]

    recent = []
    for r in reversed(rows[-12:]):
        recent.append(
            {
                "date": str(r["feature_date"]),
                "direction": r["direction"],
                "confidence": round(float(r["confidence"]) * 100, 1),
                "actual": r["actual_direction"],
                "is_correct": r["is_correct"],
                "return_pct": (
                    round(float(r["actual_return_pct"]), 2)
                    if r["actual_return_pct"] is not None
                    else None
                ),
            }
        )

    edge = (hit_rate - baseline_rate) * 100

    return {
        "empty": False,
        "latest": {
            "date": str(latest["feature_date"]),
            "direction": latest["direction"],
            "raw_score": round(float(latest["raw_score"]), 4),
            "confidence": round(float(latest["confidence"]) * 100, 1),
        },
        "stats": {
            "hit_rate": round(hit_rate * 100, 1),
            "baseline_rate": round(baseline_rate * 100, 1),
            "edge": round(edge, 1),
            "sample": n,
            "margin": round(margin * 100, 1),
            "total": len(rows),
            "within_noise": abs(edge) <= margin * 100,
        },
        "chart": chart,
        "recent": recent,
    }


def get_data() -> dict:
    """Cached accessor. At most one BigQuery query per TTL window."""

    now = time.time()
    fresh = _cache["data"] is not None and now - _cache["fetched_at"] < CACHE_TTL_SECONDS

    if fresh:
        return _cache["data"]

    try:
        _cache["data"] = build_view_model(query_performance())
        _cache["fetched_at"] = now
    except Exception as exc:
        # keep serving the last good result rather than breaking the page
        if _cache["data"] is None:
            raise
        print(f"BigQuery refresh failed, serving cached data: {exc}")

    return _cache["data"]


@app.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        get_data(),
    )


@app.get("/api/predictions")
def api_predictions():
    return JSONResponse(get_data())


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))