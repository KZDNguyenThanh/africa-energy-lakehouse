WITH years AS (
    SELECT DISTINCT year
    FROM {{ ref('silver_households') }}
)

SELECT
    ROW_NUMBER() OVER (ORDER BY year)                  AS date_key,
    year,
    CAST(FLOOR(year / 10) * 10 AS INTEGER) || 's'      AS decade,
    year - 2018                                        AS years_since_start,
    (year IN (2020, 2021))                             AS is_pandemic_year,
    CASE WHEN year <= 2020 THEN 'baseline' ELSE 'recent' END AS sdg7_period
FROM years
