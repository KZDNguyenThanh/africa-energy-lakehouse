/*
  Gold — gold_dim_geography
  Dimension bảng địa lý theo mô hình Kimball.
  Khóa tự nhiên: (country, region, is_urban, settlement_type)
*/

SELECT
    ROW_NUMBER() OVER (
        ORDER BY country, region, is_urban, settlement_type
    )                   AS geography_key,
    country,
    region,
    is_urban,
    settlement_type,
    -- Nhóm settlement cho phân tích tổng hợp
    CASE settlement_type
        WHEN 'major_city'       THEN 'urban'
        WHEN 'secondary_city'   THEN 'urban'
        WHEN 'peri_urban'       THEN 'peri_urban'
        WHEN 'rural_village'    THEN 'rural'
        WHEN 'dispersed_rural'  THEN 'rural'
        WHEN 'remote_rural'     THEN 'rural'
        ELSE 'other'
    END                 AS settlement_group
FROM (
    SELECT DISTINCT
        country,
        region,
        is_urban,
        settlement_type
    FROM {{ ref('silver_households') }}
) sub
