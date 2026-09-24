-- models/marts/mart_prediction_performance.sql
--
-- Scores every stored prediction against the price 3 days later.
-- Carries an always-bullish baseline so hit rate is never reported
-- without the number it has to beat.

{{ config(materialized='table', schema='marts') }}

with predictions as (

    select
        prediction_date,
        feature_date,
        direction,
        confidence,
        raw_score
    from {{ source('marts', 'predictions') }}

),

prices as (

    select price_date, btc_close
    from {{ source('raw', 'btc_prices') }}

),

resolved as (

    select
        p.prediction_date,
        p.feature_date,
        p.direction,
        p.confidence,
        p.raw_score,
        base.btc_close as price_at_call,
        fwd.btc_close  as price_3d_later,

        case
            when fwd.btc_close is null then null
            when fwd.btc_close > base.btc_close then 'BULLISH'
            else 'BEARISH'
        end as actual_direction

    from predictions p
    left join prices base on base.price_date = p.feature_date
    left join prices fwd  on fwd.price_date  = date_add(p.feature_date, interval 3 day)

),

scored as (

    select
        *,

        case
            when actual_direction is null then null
            when direction = actual_direction then 1
            else 0
        end as is_correct,

        case
            when actual_direction is null then null
            when actual_direction = 'BULLISH' then 1
            else 0
        end as baseline_correct,

        round(
            (price_3d_later - price_at_call) / nullif(price_at_call, 0) * 100,
            4
        ) as actual_return_pct

    from resolved

)

select
    *,

    round(
        avg(is_correct) over (
            order by feature_date
            rows between 29 preceding and current row
        ),
        4
    ) as hit_rate_30d,

    round(
        avg(baseline_correct) over (
            order by feature_date
            rows between 29 preceding and current row
        ),
        4
    ) as baseline_rate_30d

from scored
where price_at_call is not null