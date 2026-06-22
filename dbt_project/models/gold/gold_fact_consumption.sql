/*
  Gold — gold_fact_consumption
  Bảng Fact trung tâm theo mô hình Kimball (star schema).
  Mỗi dòng = 1 quan sát tiêu thụ điện của 1 hộ trong 1 năm.
  Tham chiếu đến: gold_dim_household, gold_dim_geography
*/

SELECT
    -- Surrogate keys
    s.household_id                                      AS household_key,
    g.geography_key,
    d.date_key,

    -- Degenerate dimensions (không đủ lớn để tách riêng)
    s.year,
    s.scenario,

    -- Measures chính
    s.monthly_electricity_kwh,
    s.monthly_energy_bill_usd,
    s.connection_fee_usd,
    s.hours_electricity_available,
    s.satisfaction_score,
    s.willingness_to_pay_monthly_usd,
    s.distance_to_grid_km,

    -- Derived measures
    s.appliance_count,
    -- Điểm thiết bị điện (0–6): proxy cho mức độ điện khí hóa
    (
        s.has_tv::INT            +
        s.has_fridge::INT        +
        s.has_fan::INT           +
        s.has_electric_iron::INT +
        s.has_electric_kettle::INT +
        s.has_phone_charging::INT
    )                                                   AS electric_appliance_score,

    -- Chi phí điện trên mỗi kWh (USD/kWh), tránh chia 0
    CASE
        WHEN s.monthly_electricity_kwh > 0
        THEN ROUND(s.monthly_energy_bill_usd / s.monthly_electricity_kwh, 4)
        ELSE NULL
    END                                                 AS cost_per_kwh,

    -- Flags cho dashboard lọc
    s.has_electricity_access,
    s.uses_multiple_sources,
    s.has_backup_source,
    s.grid_extension_planned,
    s.barrier_count

FROM {{ ref('silver_households') }} s
LEFT JOIN {{ ref('gold_dim_geography') }} g
    ON  s.country          = g.country
    AND s.region           = g.region
    AND s.is_urban         = g.is_urban
    AND s.settlement_type  = g.settlement_type
LEFT JOIN {{ ref('gold_dim_date') }} d
    ON  s.year             = d.year
