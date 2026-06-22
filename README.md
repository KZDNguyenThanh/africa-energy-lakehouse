# Africa Energy Lakehouse

**Đề tài:** Xây dựng hệ thống Lakehouse phân tích và dự đoán mức tiêu thụ điện năng hộ gia đình khu vực Hạ Sahara hướng tới phát triển bền vững.

**Dataset:** [electricsheepafrica/africa-synth-energy-household-electricity-access-africa-all](https://huggingface.co/datasets/electricsheepafrica/africa-synth-energy-household-electricity-access-africa-all) — 15.000 hộ, 12 quốc gia, 2018–2025. _Dữ liệu synthetic._

---

## Kiến trúc

```
Hugging Face
     │
     ▼ ingest.py
┌─────────────┐
│   BRONZE    │  bronze.households  (dữ liệu thô)
└─────────────┘
     │
     ▼ dbt
┌─────────────┐
│   SILVER    │  silver.silver_households  (đã làm sạch)
└─────────────┘
     │
     ▼ dbt
┌─────────────┐
│    GOLD     │  gold_dim_geography / gold_dim_date
│             │  gold_dim_household (SCD2, từ snapshot)
│             │  gold_fact_consumption / gold_agg_country_year
│             │  gold_ml_features / gold_predictions
└─────────────┘
     │                     │
     ▼ Colab (LightGBM)    ▼ Streamlit + Groq
   ML Model             Dashboard + GenBI
```

## Stack

| Tầng                | Công cụ                                              |
| ------------------- | ---------------------------------------------------- |
| Storage / Lakehouse | [MotherDuck](https://motherduck.com/) (DuckDB cloud) |
| Ingestion           | Python + `datasets` library                          |
| Transform           | dbt-duckdb                                           |
| ML                  | LightGBM + SHAP (Google Colab)                       |
| Dashboard           | Streamlit + Plotly                                   |
| GenBI               | Groq API (Llama 3.3-70B)                             |
| Orchestration       | GitHub Actions                                       |

## Bài toán ML

**Hồi quy (Regression):** Dự đoán `monthly_electricity_kwh` từ đặc điểm hộ gia đình.

- Chỉ huấn luyện trên hộ **có điện** và kwh > 0
- Model: LightGBM Regressor
- Metrics: MAE, RMSE, R² (so với baseline mean)
- Explainability: SHAP summary plot + force plot

## Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/<username>/africa-energy-lakehouse.git
cd africa-energy-lakehouse

# 2. Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Tạo file .env
cp .env.example .env
# Điền MOTHERDUCK_TOKEN, HF_TOKEN, GROQ_API_KEY vào .env

# 5. Ingest dữ liệu vào Bronze
python ingest.py

# 6. Chạy dbt (Silver + Gold)
cd dbt_project
dbt run --profiles-dir .
cd ..

# 7. Chạy ML notebook trên Google Colab
# Upload ml/train_model.ipynb lên Colab, điền token, Run All

# 8. Chạy dashboard local
streamlit run app/streamlit_app.py
```

## Cấu trúc thư mục

```
africa-energy-lakehouse/
├── ingest.py                    # Bronze ingestion
├── requirements.txt
├── .env.example
├── dbt_project/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── snapshots/
│   │   └── household_snapshot.sql   # SCD Type 2
│   └── models/
│       ├── silver/
│       │   └── silver_households.sql
│       └── gold/
│           ├── gold_dim_geography.sql
│           ├── gold_dim_date.sql
│           ├── gold_dim_household.sql
│           ├── gold_fact_consumption.sql
│           ├── gold_agg_country_year.sql
│           └── gold_ml_features.sql
├── ml/
│   └── train_model.ipynb
├── app/
│   └── streamlit_app.py
└── .github/workflows/
    └── pipeline.yml
```
