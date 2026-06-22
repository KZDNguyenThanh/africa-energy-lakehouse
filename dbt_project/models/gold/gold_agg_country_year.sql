/*
  Gold — gold_agg_country_year
  Bảng tổng hợp theo quốc gia × năm × scenario.
  Phục vụ trực tiếp cho Streamlit dashboard (không cần query nặng runtime).
*/

SELECT
    country,
    year,
    scenario,
    is_urban,
    settlement_group,

    -- Volume
    COUNT(*)                                                AS n_households,
    SUM(CASE WHEN f.has_electricity_access THEN 1 ELSE 0 END) AS n_with_electricity,

    -- Access rate
    ROUND(
        AVG(CASE WHEN f.has_electricity_access THEN 1.0 ELSE 0.0 END) * 100, 1
    )                                                       AS access_rate_pct,

    -- Tiêu thụ (chỉ hộ có điện)
    ROUND(AVG(CASE WHEN monthly_electricity_kwh > 0
                   THEN monthly_electricity_kwh END), 2)    AS avg_kwh_electrified,
    ROUND(MAX(monthly_electricity_kwh), 2)                  AS max_kwh,
    ROUND(MIN(CASE WHEN monthly_electricity_kwh > 0
                   THEN monthly_electricity_kwh END), 2)    AS min_kwh_electrified,

    -- Chi phí
    ROUND(AVG(CASE WHEN monthly_energy_bill_usd > 0
                   THEN monthly_energy_bill_usd END), 2)    AS avg_bill_usd,
    ROUND(AVG(willingness_to_pay_monthly_usd), 2)           AS avg_wtp_usd,

    -- Chất lượng điện
    ROUND(AVG(CASE WHEN hours_electricity_available > 0
                   THEN hours_electricity_available END), 1) AS avg_hours_available,
    ROUND(AVG(satisfaction_score), 1)                       AS avg_satisfaction,

    -- Thiết bị
    ROUND(AVG(electric_appliance_score), 2)                 AS avg_appliance_score,

    -- Rào cản
    ROUND(AVG(barrier_count), 2)                            AS avg_barrier_count

FROM {{ ref('gold_fact_consumption') }} f
LEFT JOIN {{ ref('gold_dim_geography') }} g USING (geography_key)

GROUP BY country, year, scenario, is_urban, settlement_group
ORDER BY country, year
