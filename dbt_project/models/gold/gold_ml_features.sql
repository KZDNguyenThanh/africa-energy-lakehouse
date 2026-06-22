/*
  Gold — gold_ml_features
  Bảng wide (phẳng) phục vụ riêng cho huấn luyện ML regression.

  Quyết định thiết kế quan trọng:
    - CHỈ lấy hộ CÓ điện VÀ monthly_electricity_kwh > 0
    - Lý do: hộ không có điện luôn có kwh = 0 → nếu để vào,
      model chỉ học phân biệt có/không điện thay vì học
      cường độ tiêu thụ thực sự.
    - Biến mục tiêu: monthly_electricity_kwh (target_kwh)
*/

SELECT
    household_id,

    -- Geographic features
    country,
    region,
    is_urban,
    settlement_type,

    -- Household features
    household_size,
    income_quintile,
    education_head,
    connection_type,
    connection_quality,
    years_connected,

    -- Energy access features
    hours_electricity_available,
    distance_to_grid_km,
    appliance_count,

    -- Appliance flags (boolean → int trong notebook)
    has_phone_charging,
    has_radio,
    has_tv,
    has_fridge,
    has_fan,
    has_electric_iron,
    has_electric_kettle,

    -- Behavioral features
    satisfaction_score,
    willingness_to_pay_monthly_usd,
    primary_cooking_fuel,
    uses_multiple_sources,
    has_backup_source,
    barrier_count,

    -- Context
    scenario,
    year,

    -- TARGET VARIABLE
    monthly_electricity_kwh     AS target_kwh

FROM {{ ref('silver_households') }}
WHERE has_electricity_access = TRUE
  AND monthly_electricity_kwh > 0
