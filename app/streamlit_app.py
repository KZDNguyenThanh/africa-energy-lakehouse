"""
Streamlit Dashboard — Africa Energy Lakehouse
Hiển thị phân tích tiêu thụ điện từ Gold layer + serving dự đoán ML + GenBI với Groq.
"""

import os
import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

# ─── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Africa Energy Lakehouse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _secret(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return ""


MD_TOKEN   = _secret("MOTHERDUCK_TOKEN")
GROQ_KEY   = _secret("GROQ_API_KEY")
DB_NAME    = "energy_lakehouse"

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load(query: str, params: list | None = None) -> pd.DataFrame:
    con = duckdb.connect(f"md:{DB_NAME}?motherduck_token={MD_TOKEN}")
    df = con.execute(query, params or []).fetchdf()
    con.close()
    return df


def check_token():
    if not MD_TOKEN:
        st.error("MOTHERDUCK_TOKEN chưa được set. Kiểm tra file .env hoặc Streamlit Secrets.")
        st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Bộ lọc")
    check_token()

    agg = load("SELECT DISTINCT country, year, scenario, is_urban FROM gold.gold_agg_country_year ORDER BY country")

    selected_countries = st.multiselect(
        "Quốc gia",
        sorted(agg["country"].unique()),
        default=sorted(agg["country"].unique())[:6]
    )
    selected_years = st.slider(
        "Năm",
        int(agg["year"].min()),
        int(agg["year"].max()),
        (int(agg["year"].min()), int(agg["year"].max()))
    )
    selected_scenario = st.selectbox(
        "Kịch bản",
        ["Tất cả"] + sorted(agg["scenario"].unique())
    )
    urban_filter = st.radio("Khu vực", ["Tất cả", "Đô thị", "Nông thôn"])

    st.divider()
    st.caption("Dữ liệu: Africa Synth Energy — Hugging Face\nKiến trúc: Lakehouse Medallion (Bronze → Silver → Gold)")

# ─── Load data với filter (parameterized — tránh SQL injection) ───────────────
country_choice = selected_countries or [""]
placeholders = ", ".join("?" for _ in country_choice)

agg_clauses = [
    f"country IN ({placeholders})",
    "year BETWEEN ? AND ?",
]
agg_params: list = [*country_choice, selected_years[0], selected_years[1]]

if selected_scenario != "Tất cả":
    agg_clauses.append("scenario = ?")
    agg_params.append(selected_scenario)
if urban_filter != "Tất cả":
    agg_clauses.append("is_urban = ?")
    agg_params.append(urban_filter == "Đô thị")

agg_df = load(
    f"""
    SELECT * FROM gold.gold_agg_country_year
    WHERE {' AND '.join(agg_clauses)}
    ORDER BY country, year
    """,
    agg_params,
)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("⚡ Hệ thống Lakehouse — Phân tích tiêu thụ điện năng hộ gia đình")
st.caption("Dự đoán mức tiêu thụ điện năng hộ gia đình khu vực Hạ Sahara | 12 quốc gia | 2018–2025")

# ─── KPI metrics ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
if not agg_df.empty:
    col1.metric("Tổng hộ quan sát",   f"{agg_df['n_households'].sum():,}")
    col2.metric("Hộ có điện",         f"{agg_df['n_with_electricity'].sum():,}")
    col3.metric("Tỷ lệ có điện TB",   f"{agg_df['access_rate_pct'].mean():.1f}%")
    col4.metric("TB tiêu thụ (kWh)",  f"{agg_df['avg_kwh_electrified'].mean():.1f}")

st.divider()

# ─── Tab layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Tổng quan tiêu thụ",
    "Tiếp cận điện",
    "Dự đoán ML",
    "GenBI — Nhận định AI"
])

# ── Tab 1: Tiêu thụ ───────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tiêu thụ điện TB theo năm (kWh/tháng)")
        pivot = agg_df.groupby(["year", "country"])["avg_kwh_electrified"].mean().reset_index()
        fig = px.line(pivot, x="year", y="avg_kwh_electrified", color="country",
                      markers=True, labels={"avg_kwh_electrified": "kWh/tháng", "year": "Năm"})
        fig.update_layout(legend_title="Quốc gia", height=350)
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Phân phối kWh theo quốc gia")
        raw = load(
            f"""
            SELECT g.country, f.monthly_electricity_kwh
            FROM gold.gold_fact_consumption f
            JOIN gold.gold_dim_geography g USING (geography_key)
            WHERE g.country IN ({placeholders})
              AND f.monthly_electricity_kwh > 0
              AND f.year BETWEEN ? AND ?
            """,
            [*country_choice, selected_years[0], selected_years[1]],
        )
        fig2 = px.box(raw, x="country", y="monthly_electricity_kwh",
                      color="country", labels={"monthly_electricity_kwh": "kWh/tháng"})
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Số giờ điện khả dụng TB / ngày")
    hrs = agg_df.groupby("country")["avg_hours_available"].mean().sort_values(ascending=True).reset_index()
    fig3 = px.bar(hrs, x="avg_hours_available", y="country", orientation="h",
                  labels={"avg_hours_available": "Giờ/ngày", "country": ""},
                  color="avg_hours_available", color_continuous_scale="Blues")
    fig3.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig3, width='stretch')

# ── Tab 2: Tiếp cận điện ──────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tỷ lệ tiếp cận điện theo quốc gia (%)")
        acc = agg_df.groupby("country")["access_rate_pct"].mean().sort_values(ascending=False).reset_index()
        fig4 = px.bar(acc, x="country", y="access_rate_pct",
                      color="access_rate_pct", color_continuous_scale="Greens",
                      labels={"access_rate_pct": "Tỷ lệ (%)", "country": ""})
        fig4.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig4, width='stretch')

    with c2:
        st.subheader("Tỷ lệ tiếp cận điện theo năm")
        acc_yr = agg_df.groupby(["year", "country"])["access_rate_pct"].mean().reset_index()
        fig5 = px.line(acc_yr, x="year", y="access_rate_pct", color="country",
                       markers=True, labels={"access_rate_pct": "Tỷ lệ (%)", "year": "Năm"})
        fig5.update_layout(height=350, legend_title="Quốc gia")
        st.plotly_chart(fig5, width='stretch')

    st.subheader("Điểm thiết bị điện trung bình (0–6)")
    app = agg_df.groupby(["country", "is_urban"])["avg_appliance_score"].mean().reset_index()
    app["khu_vuc"] = app["is_urban"].map({True: "Đô thị", False: "Nông thôn"})
    fig6 = px.bar(app, x="country", y="avg_appliance_score", color="khu_vuc",
                  barmode="group", labels={"avg_appliance_score": "Điểm TB", "country": ""})
    fig6.update_layout(height=320, legend_title="Khu vực")
    st.plotly_chart(fig6, width='stretch')

# ── Tab 3: ML Predictions ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Kết quả dự đoán mức tiêu thụ điện (LightGBM Regression)")

    try:
        pred_df = load("""
            SELECT actual_kwh, predicted_kwh, error_kwh,
                   country, income_quintile, settlement_type
            FROM gold.gold_predictions
            LIMIT 2000
        """)

        if pred_df.empty:
            st.info("Chưa có dữ liệu dự đoán. Chạy notebook ml/train_model.ipynb trước.")
        else:
            mae  = pred_df["error_kwh"].abs().mean()
            rmse = (pred_df["error_kwh"] ** 2).mean() ** 0.5
            ss_res = ((pred_df["actual_kwh"] - pred_df["predicted_kwh"]) ** 2).sum()
            ss_tot = ((pred_df["actual_kwh"] - pred_df["actual_kwh"].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE (kWh)", f"{mae:.2f}")
            m2.metric("RMSE (kWh)", f"{rmse:.2f}")
            m3.metric("R²", f"{r2:.4f}")

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Predicted vs Actual")
                fig7 = px.scatter(pred_df, x="actual_kwh", y="predicted_kwh",
                                  opacity=0.4, color="country",
                                  labels={"actual_kwh": "Thực tế (kWh)", "predicted_kwh": "Dự đoán (kWh)"})
                max_val = max(pred_df["actual_kwh"].max(), pred_df["predicted_kwh"].max())
                fig7.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                          mode="lines", name="Perfect",
                                          line=dict(color="red", dash="dash")))
                fig7.update_layout(height=380)
                st.plotly_chart(fig7, width='stretch')

            with c2:
                st.caption("Phân phối sai số dự đoán")
                fig8 = px.histogram(pred_df, x="error_kwh", nbins=40,
                                    labels={"error_kwh": "Sai số (kWh)"})
                fig8.add_vline(x=0, line_dash="dash", line_color="red")
                fig8.update_layout(height=380)
                st.plotly_chart(fig8, width='stretch')

    except Exception as e:
        st.info(f"Bảng gold_predictions chưa tồn tại. Chạy notebook ML trước.\n\n`{e}`")

# ── Tab 4: GenBI ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Trợ lý phân tích AI (Groq — Llama 3)")

    if not GROQ_KEY:
        st.warning("GROQ_API_KEY chưa được set. Thêm vào .env hoặc Streamlit Secrets.")
    else:
        summary_data = agg_df.groupby("country").agg(
            access_rate=("access_rate_pct", "mean"),
            avg_kwh=("avg_kwh_electrified", "mean"),
            avg_bill=("avg_bill_usd", "mean"),
            avg_hours=("avg_hours_available", "mean"),
            avg_satisfaction=("avg_satisfaction", "mean")
        ).round(2)

        st.dataframe(summary_data, width='stretch')

        preset_questions = [
            "Phân tích top 3 quốc gia có mức tiêu thụ điện cao nhất và lý do",
            "Đề xuất 3 ưu tiên chính sách để cải thiện tiếp cận điện khu vực nông thôn",
            "So sánh hiệu quả chi phí điện giữa các quốc gia và khuyến nghị",
            "Phân tích mối quan hệ giữa thu nhập và tiêu thụ điện, hàm ý chính sách"
        ]
        selected_q = st.selectbox("Câu hỏi gợi ý", ["-- Tự nhập --"] + preset_questions)
        user_prompt = st.text_area(
            "Hoặc nhập câu hỏi của bạn:",
            value=selected_q if selected_q != "-- Tự nhập --" else "",
            height=80
        )

        if st.button("Tạo phân tích", type="primary") and user_prompt:
            with st.spinner("Đang phân tích..."):
                context = f"""Dữ liệu tổng hợp tiêu thụ điện hộ gia đình khu vực Hạ Sahara
(access_rate: % hộ có điện, avg_kwh: kWh/tháng, avg_bill: USD/tháng,
 avg_hours: giờ điện khả dụng/ngày, avg_satisfaction: điểm hài lòng 0–10):

{summary_data.to_string()}

Dữ liệu synthetic mô phỏng từ 12 quốc gia, 2018–2025.
"""
                try:
                    client = Groq(api_key=GROQ_KEY)
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "Bạn là chuyên gia phân tích năng lượng và chính sách phát triển bền vững. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng, ngắn gọn và thực tiễn."},
                            {"role": "user", "content": f"{context}\n\nCâu hỏi: {user_prompt}"}
                        ],
                        max_tokens=800
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Lỗi Groq API: {e}")
