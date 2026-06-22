SELECT
    ROW_NUMBER() OVER (ORDER BY household_id, dbt_valid_from) AS household_sk,
    household_id                AS household_key,
    household_size,
    income_quintile,
    education_head,
    has_electricity_access,
    connection_type,
    connection_quality,
    years_connected,
    appliance_count,
    CASE income_quintile
        WHEN 1 THEN 'very_low'
        WHEN 2 THEN 'low'
        WHEN 3 THEN 'middle'
        WHEN 4 THEN 'high'
        WHEN 5 THEN 'very_high'
    END                         AS income_group,
    CASE
        WHEN connection_type = 'grid_direct'    THEN 'grid'
        WHEN connection_type = 'grid_shared'    THEN 'grid'
        WHEN connection_type = 'mini_grid'      THEN 'off_grid'
        WHEN connection_type = 'off_grid_solar' THEN 'off_grid'
        WHEN connection_type = 'generator'      THEN 'off_grid'
        ELSE 'none'
    END                         AS energy_source_category,
    dbt_valid_from              AS valid_from,
    dbt_valid_to                AS valid_to,
    (dbt_valid_to IS NULL)      AS is_current
FROM {{ ref('household_snapshot') }}
