{% snapshot household_snapshot %}
{{
    config(
        target_schema='snapshots',
        unique_key='household_id',
        strategy='check',
        check_cols=[
            'household_size',
            'income_quintile',
            'education_head',
            'has_electricity_access',
            'connection_type',
            'connection_quality',
            'years_connected',
            'appliance_count'
        ],
        invalidate_hard_deletes=True
    )
}}

SELECT
    household_id,
    household_size,
    income_quintile,
    education_head,
    has_electricity_access,
    connection_type,
    connection_quality,
    years_connected,
    appliance_count
FROM {{ ref('silver_households') }}

{% endsnapshot %}
