-- models/staging/stg_daily_sentiment.sql
--
-- published_raw arrives in two shapes: plain dates from the historical
-- CSV backfill, and RFC-822 strings from Google News RSS
-- ("Tue, 22 Jul 2026 14:30:00 GMT"). Parse both explicitly rather than
-- letting one source silently drop out.

{{ config(materialized='view', schema='staging') }}

with headlines as (

    select
        h.link,
        coalesce(
            safe_cast(h.published_raw as date),
            date(safe_cast(h.published_raw as timestamp)),
            date(safe.parse_timestamp('%a, %d %b %Y %H:%M:%S %Z', h.published_raw)),
            safe.parse_date(
                '%d %b %Y',
                regexp_extract(h.published_raw, r'(\d{1,2} [A-Za-z]{3} \d{4})')
            )
        ) as headline_date,
        s.sentiment_score

    from {{ source('raw', 'crypto_headlines') }} h
    inner join {{ source('raw', 'headline_sentiments') }} s
        on h.link = s.link

),

daily as (

    select
        headline_date        as sentiment_date,
        avg(sentiment_score) as sentiment_score,
        count(*)             as headline_count

    from headlines
    where headline_date is not null
    group by headline_date

)

select * from daily