/*
  Silver — silver_households
  Làm sạch dữ liệu thô từ Bronze:
    - Loại bản ghi trùng (giữ bản mới nhất theo _ingested_at)
    - Chuẩn hóa kiểu dữ liệu (INTEGER, FLOAT, BOOLEAN)
    - Xử lý NULL bằng COALESCE
    - Loại giá trị vô lý (household_size, year ngoài range)
    - Tách cột barriers_to_access thành mảng để phân tích
*/

WITH src AS (
    SELECT * FROM bronze.households
),

dedup AS (
    -- household_id phải là duy nhất; phòng trường hợp ingest nhiều lần
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY household_id
            ORDER BY _ingested_at DESC
        ) AS _rn
    FROM src
),

cleaned AS (
    SELECT
        household_id,
        TRIM(country)                                       AS country,
        CAST(year AS INTEGER)                               AS year,
        CAST(is_urban AS BOOLEAN)                           AS is_urban,
        TRIM(settlement_type)                               AS settlement_type,
        CAST(household_size AS INTEGER)                     AS household_size,
        CAST(income_quintile AS INTEGER)                    AS income_quintile,
        COALESCE(NULLIF(TRIM(education_head), ''), 'unknown')   AS education_head,
        CAST(has_electricity_access AS BOOLEAN)             AS has_electricity_access,
        COALESCE(NULLIF(TRIM(connection_type), ''), 'none')     AS connection_type,
        COALESCE(NULLIF(TRIM(connection_quality), ''), 'none')  AS connection_quality,
        COALESCE(CAST(hours_electricity_available AS FLOAT), 0) AS hours_electricity_available,
        COALESCE(NULLIF(TRIM(primary_lighting_source), ''), 'unknown') AS primary_lighting_source,
        COALESCE(NULLIF(TRIM(primary_cooking_fuel), ''), 'unknown')    AS primary_cooking_fuel,
        COALESCE(CAST(monthly_electricity_kwh AS FLOAT), 0)    AS monthly_electricity_kwh,
        COALESCE(CAST(monthly_energy_bill_usd AS FLOAT), 0)    AS monthly_energy_bill_usd,
        COALESCE(CAST(connection_fee_usd AS FLOAT), 0)         AS connection_fee_usd,
        COALESCE(CAST(distance_to_grid_km AS FLOAT), 0)        AS distance_to_grid_km,
        COALESCE(CAST(years_connected AS INTEGER), 0)          AS years_connected,
        COALESCE(CAST(appliance_count AS INTEGER), 0)          AS appliance_count,
        CAST(has_phone_charging AS BOOLEAN)                 AS has_phone_charging,
        CAST(has_radio AS BOOLEAN)                          AS has_radio,
        CAST(has_tv AS BOOLEAN)                             AS has_tv,
        CAST(has_fridge AS BOOLEAN)                         AS has_fridge,
        CAST(has_fan AS BOOLEAN)                            AS has_fan,
        CAST(has_electric_iron AS BOOLEAN)                  AS has_electric_iron,
        CAST(has_electric_kettle AS BOOLEAN)                AS has_electric_kettle,
        COALESCE(CAST(satisfaction_score AS INTEGER), 0)       AS satisfaction_score,
        COALESCE(CAST(willingness_to_pay_monthly_usd AS FLOAT), 0) AS willingness_to_pay_monthly_usd,
        -- Tách pipe-separated barriers thành LIST để dễ phân tích sau
        CASE
            WHEN TRIM(barriers_to_access) IN ('none', '', 'null')
            THEN []
            ELSE STRING_SPLIT(TRIM(barriers_to_access), '|')
        END                                                 AS barriers_list,
        TRIM(barriers_to_access)                            AS barriers_raw,
        -- Đếm số rào cản (0 = không có rào cản)
        CASE
            WHEN TRIM(barriers_to_access) IN ('none', '', 'null') THEN 0
            ELSE ARRAY_LENGTH(STRING_SPLIT(TRIM(barriers_to_access), '|'))
        END                                                 AS barrier_count,
        CAST(uses_multiple_sources AS BOOLEAN)              AS uses_multiple_sources,
        CAST(has_backup_source AS BOOLEAN)                  AS has_backup_source,
        CAST(grid_extension_planned AS BOOLEAN)             AS grid_extension_planned,
        TRIM(region)                                        AS region,
        TRIM(scenario)                                      AS scenario,
        _ingested_at
    FROM dedup
    WHERE _rn = 1
      AND CAST(year AS INTEGER) BETWEEN 2018 AND 2025
      AND CAST(household_size AS INTEGER) BETWEEN 1 AND 20
      AND CAST(income_quintile AS INTEGER) BETWEEN 1 AND 5
)

SELECT * FROM cleaned
