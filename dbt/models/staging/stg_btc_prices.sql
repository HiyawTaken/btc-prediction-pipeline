-- models/staging/stg_btc_prices.sql

{{ config(materialized='view', schema='staging') }}

with source as (

    select * from {{ source('raw', 'btc_prices') }}

),

cleaned as (

    select
        price_date,
        btc_high,
        btc_low,
        btc_close,
        btc_volume,
        spy_close,
        dxy_close,

        ln(btc_close / nullif(lag(btc_close) over (order by price_date), 0)) as btc_log_return,
        ln(spy_close / nullif(lag(spy_close) over (order by price_date), 0)) as spy_log_return,
        ln(dxy_close / nullif(lag(dxy_close) over (order by price_date), 0)) as dxy_log_return

    from source
    where btc_close > 0

)

select * from cleaned