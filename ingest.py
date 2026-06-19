import os
import sys
import duckdb
import pandas as pd
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

MD_TOKEN = os.environ.get("MOTHERDUCK_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

DATASET_ID = "electricsheepafrica/africa-synth-energy-household-electricity-access-africa-all"
DB_NAME    = "energy_lakehouse"

CSV_FILES = [
    "household_electricity_access_low_burden.csv",
    "household_electricity_access_moderate_burden.csv",
    "household_electricity_access_high_burden.csv",
]


def get_connection():
    if not MD_TOKEN:
        print("[ERROR] MOTHERDUCK_TOKEN chưa được set trong .env")
        sys.exit(1)
    con = duckdb.connect(f"md:?motherduck_token={MD_TOKEN}")
    con.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
    con.execute(f"USE {DB_NAME};")
    return con


def ingest():
    print(f"[1/3] Đang tải dataset '{DATASET_ID}' từ Hugging Face...")
    frames = []
    for fname in CSV_FILES:
        path = hf_hub_download(DATASET_ID, fname, repo_type="dataset", token=HF_TOKEN)
        part = pd.read_csv(path)
        print(f"      → {fname}: {len(part):,} dòng")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    print(f"      → Tổng: {len(df):,} dòng, {len(df.columns)} cột")

    print("[2/3] Kết nối MotherDuck...")
    con = get_connection()

    print("[3/3] Nạp vào bronze.households...")
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    con.register("_df_staging", df)
    con.execute("""
        CREATE OR REPLACE TABLE bronze.households AS
        SELECT
            *,
            current_timestamp AS _ingested_at,
            'huggingface'      AS _source,
            'africa-synth-energy-household-electricity-access-africa-all' AS _dataset_id
        FROM _df_staging;
    """)

    n = con.execute("SELECT count(*) FROM bronze.households").fetchone()[0]
    print(f"\n[OK] Bronze ingestion hoàn tất: {n:,} dòng trong bronze.households")
    con.close()


if __name__ == "__main__":
    ingest()
