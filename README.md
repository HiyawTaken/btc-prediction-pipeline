```
btc-pipeline/
├── .env                        # never committed
├── .gitignore
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.airflow
├── Dockerfile.api
├── render.yaml
│
├── dags/
│   └── btc_daily_pipeline.py   # Airflow DAG
│
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml            # never committed
│   └── models/
│       ├── staging/
│       │   ├── stg_btc_prices.sql
│       │   └── stg_daily_sentiment.sql
│       └── marts/
│           ├── mart_btc_features.sql
│           └── mart_predictions.sql
│
├── src/
│   ├── ingestion/
│   │   ├── ingest_prices.py
│   │   ├── ingest_news.py
│   │   └── label_sentiment.py
│   ├── inference/
│   │   ├── lstm_numpy.py
│   │   └── predict.py
│   └── api/
│       ├── app.py
│       └── templates/
│           └── index.html
│
└── models/
    ├── lstm_best.npz
    └── scaler.json
```