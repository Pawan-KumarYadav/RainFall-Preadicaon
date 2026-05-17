"""
╔══════════════════════════════════════════════════════════════════╗
║          🌧️  RainCast AI — Streamlit Frontend                    ║
╚══════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import time
import io

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RainCast AI",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── API Config ───────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & background ──────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar gradient ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1e3d 0%, #1a3560 50%, #0d2b55 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #4facfe !important;
}

/* ── Top header banner ────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f1e3d 0%, #1565c0 50%, #0288d1 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-banner h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
.hero-banner p  { font-size: 1.05rem; margin: 0.3rem 0 0; opacity: 0.85; }

/* ── Result card ──────────────────────────────────────── */
.result-rain {
    background: linear-gradient(135deg, #1565c0, #0288d1);
    border-radius: 16px;
    padding: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(21,101,192,0.35);
}
.result-norain {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border-radius: 16px;
    padding: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(27,94,32,0.35);
}
.result-label { font-size: 3rem; margin-bottom: 0.3rem; }
.result-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
.result-sub   { font-size: 1rem; opacity: 0.85; margin-top: 0.3rem; }

/* ── Metric cards ─────────────────────────────────────── */
.metric-card {
    background: #f8faff;
    border: 1px solid #dbe8ff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .val  { font-size: 1.8rem; font-weight: 700; color: #1565c0; }
.metric-card .lbl  { font-size: 0.8rem; color: #607d8b; margin-top: 0.2rem; }

/* ── Section title ────────────────────────────────────── */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1565c0;
    border-left: 4px solid #1565c0;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

/* ── Badge ────────────────────────────────────────────── */
.badge-high     { background:#e8f5e9; color:#2e7d32; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.badge-moderate { background:#fff8e1; color:#f57f17; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.badge-low      { background:#ffebee; color:#b71c1c; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }

/* ── Model card ───────────────────────────────────────── */
.model-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Predict button ───────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0288d1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2.5rem;
    font-size: 1.05rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.2s;
    cursor: pointer;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #0d47a1, #0277bd);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(21,101,192,0.4);
}

/* ── Tabs ─────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}

/* ── Upload section ───────────────────────────────────── */
.uploadedFile { border: 2px dashed #1565c0 !important; border-radius: 12px !important; }

/* ── Footer ───────────────────────────────────────────── */
footer.raincast-footer {
    text-align: center;
    font-size: 0.82rem;
    color: #90a4ae;
    margin-top: 3rem;
    padding: 1rem 0;
    border-top: 1px solid #eceff1;
}
</style>
""", unsafe_allow_html=True)

# ─── API Helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def get_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.json() if r.ok else None
    except:
        return None

@st.cache_data(ttl=60)
def get_model_info():
    try:
        r = requests.get(f"{API_BASE}/model-info", timeout=4)
        return r.json() if r.ok else None
    except:
        return None

def call_predict(payload: dict):
    return requests.post(f"{API_BASE}/predict", json=payload, timeout=10)

def call_predict_batch(records: list):
    return requests.post(f"{API_BASE}/predict/batch", json={"records": records}, timeout=30)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌧️ RainCast AI")
    st.markdown("---")

    health = get_health()
    if health and health.get("models_loaded"):
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.info("Start the backend:\n```\nuvicorn main:app --reload\n```")

    st.markdown("### ⚙️ Input Parameters")

    pressure = st.slider("🌡️ Pressure (hPa)",    900.0,  1100.0, 1015.0, 0.1)
    temperature = st.slider("🌡️ Temperature (°C)", -20.0,   60.0,   22.0, 0.1)
    dewpoint = st.slider("💧 Dew Point (°C)",    -30.0,   40.0,   14.0, 0.1)
    humidity = st.slider("💦 Humidity (%)",         0,    100,     75,   1)
    cloud    = st.slider("☁️ Cloud Cover (oktas)",  0,      8,      5,   1)
    sunshine = st.slider("☀️ Sunshine (hrs)",       0.0,   24.0,    3.0, 0.1)
    winddirection = st.slider("🧭 Wind Direction (°)", 0, 360,   180,   1)
    windspeed = st.slider("💨 Wind Speed (km/h)",   0.0,  200.0,  25.0, 0.1)

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Rainfall", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.caption(
        "Ensemble of Logistic Regression, XGBoost & SVC models trained on "
        "historical meteorological data to predict daily rainfall."
    )

# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div style="font-size:3.5rem; line-height:1;">🌧️</div>
  <div>
    <h1>RainCast AI</h1>
    <p>Advanced Rainfall Prediction • Ensemble Machine Learning • Real-Time Forecasting</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_explore, tab_info = st.tabs([
    "🔮 Single Prediction", "📦 Batch Prediction", "📊 Data Explorer", "ℹ️ Model Info"
])

# ══════════════════════════ TAB 1: SINGLE PREDICTION ══════════════════════════
with tab_predict:
    payload = dict(
        pressure=pressure, temperature=temperature, dewpoint=dewpoint,
        humidity=humidity, cloud=cloud, sunshine=sunshine,
        winddirection=winddirection, windspeed=windspeed,
    )

    if predict_clicked:
        if not (health and health.get("models_loaded")):
            st.error("❌ Cannot reach API. Make sure the FastAPI backend is running.")
        else:
            with st.spinner("🤖 Running ensemble inference…"):
                t0 = time.time()
                try:
                    resp = call_predict(payload)
                    resp.raise_for_status()
                    result = resp.json()
                    elapsed_ms = round((time.time() - t0) * 1000, 1)

                    # ── Result card ──────────────────────────────────────────
                    is_rain = result["final_prediction"] == "Rain"
                    card_class = "result-rain" if is_rain else "result-norain"
                    emoji = "🌧️" if is_rain else "☀️"
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="result-label">{emoji}</div>
                        <p class="result-title">{result['final_prediction']}</p>
                        <p class="result-sub">
                            Rain Probability: {result['rain_probability']}% &nbsp;|&nbsp;
                            Confidence: {result['confidence_level']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Probability Gauge ────────────────────────────────────
                    col_gauge, col_bar = st.columns(2)
                    with col_gauge:
                        st.markdown('<div class="section-title">🌡️ Rain Probability Gauge</div>', unsafe_allow_html=True)
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=result["rain_probability"],
                            number={"suffix": "%", "font": {"size": 36, "color": "#1565c0"}},
                            delta={"reference": 50, "valueformat": ".1f"},
                            gauge={
                                "axis": {"range": [0, 100], "tickfont": {"size": 12}},
                                "bar": {"color": "#1565c0" if is_rain else "#2e7d32", "thickness": 0.3},
                                "steps": [
                                    {"range": [0, 40],  "color": "#e8f5e9"},
                                    {"range": [40, 60], "color": "#fff8e1"},
                                    {"range": [60, 100],"color": "#e3f2fd"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 3},
                                    "thickness": 0.8,
                                    "value": 50,
                                },
                            },
                        ))
                        fig_gauge.update_layout(
                            height=280, margin=dict(l=30, r=30, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col_bar:
                        st.markdown('<div class="section-title">📊 Probability Breakdown</div>', unsafe_allow_html=True)
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            x=["No Rain", "Rain"],
                            y=[result["no_rain_probability"], result["rain_probability"]],
                            marker_color=["#2e7d32", "#1565c0"],
                            text=[f"{result['no_rain_probability']}%", f"{result['rain_probability']}%"],
                            textposition="outside",
                            textfont=dict(size=14, color="#333"),
                        ))
                        fig_bar.update_layout(
                            height=280, yaxis_range=[0, 115],
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=10, t=10, b=10),
                            showlegend=False,
                            xaxis=dict(tickfont=dict(size=14)),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # ── Model-level breakdown ────────────────────────────────
                    st.markdown('<div class="section-title">🤖 Individual Model Votes</div>', unsafe_allow_html=True)
                    cols = st.columns(len(result["individual_models"]))
                    for col, m in zip(cols, result["individual_models"]):
                        with col:
                            icon = "🌧️" if m["prediction"] == "Rain" else "☀️"
                            color = "#1565c0" if m["prediction"] == "Rain" else "#2e7d32"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size:1.6rem">{icon}</div>
                                <div class="val" style="color:{color}">{m['confidence']}%</div>
                                <div class="lbl">{m['model_name']}</div>
                                <div style="font-size:0.8rem;font-weight:600;color:{color}">{m['prediction']}</div>
                            </div>""", unsafe_allow_html=True)

                    # ── Radar chart of input features ────────────────────────
                    st.markdown('<div class="section-title">🕸️ Input Feature Profile</div>', unsafe_allow_html=True)
                    feat_labels = ["Pressure", "Temp", "Dewpoint", "Humidity", "Cloud", "Sunshine", "Wind Dir", "Wind Spd"]
                    feat_vals   = [
                        pressure, temperature + 20, dewpoint + 30,  # normalise negatives
                        humidity, cloud * 12.5, sunshine * 4.17,
                        winddirection / 3.6, windspeed,
                    ]
                    # clamp to 0-100
                    feat_vals_norm = [min(max(v, 0), 100) for v in feat_vals]
                    fig_radar = go.Figure(go.Scatterpolar(
                        r=feat_vals_norm + [feat_vals_norm[0]],
                        theta=feat_labels + [feat_labels[0]],
                        fill="toself",
                        fillcolor="rgba(21,101,192,0.15)",
                        line=dict(color="#1565c0", width=2),
                    ))
                    fig_radar.update_layout(
                        height=340, polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=True, range=[0, 100]),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=30, r=30, t=30, b=30),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                    # ── Meta info ────────────────────────────────────────────
                    st.caption(
                        f"Request ID: `{result['request_id']}` • "
                        f"Inference: {result['inference_time_ms']} ms • "
                        f"Timestamp: {result['timestamp']}"
                    )

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Is the FastAPI server running on port 8000?")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        # ── Placeholder state ────────────────────────────────────────────────
        st.info("👈 Adjust the weather parameters in the sidebar and click **Predict Rainfall**.")
        col1, col2, col3, col4 = st.columns(4)
        for c, lbl, val, unit in zip(
            [col1, col2, col3, col4],
            ["Pressure", "Temperature", "Humidity", "Wind Speed"],
            [pressure, temperature, humidity, windspeed],
            ["hPa", "°C", "%", "km/h"],
        ):
            with c:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="val">{val}</div>
                    <div class="lbl">{lbl} ({unit})</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════ TAB 2: BATCH PREDICTION ═══════════════════════════
with tab_batch:
    st.markdown('<div class="section-title">📦 Batch Prediction via CSV Upload</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a CSV file with the columns:
    `pressure, temperature, dewpoint, humidity, cloud, sunshine, winddirection, windspeed`
    """)

    # Sample CSV download
    sample_df = pd.DataFrame({
        "pressure":     [1015.0, 1008.5, 1022.3],
        "temperature":  [22.5,   28.0,   18.3],
        "dewpoint":     [14.0,   22.5,   10.1],
        "humidity":     [75,     88,     55],
        "cloud":        [5,      7,      2],
        "sunshine":     [3.2,    0.5,    9.0],
        "winddirection":[180,    220,    90],
        "windspeed":    [25.0,   35.0,   12.0],
    })
    csv_sample = sample_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample CSV", csv_sample, "sample_input.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.write(f"**{len(df_upload)} records loaded**")
        st.dataframe(df_upload.head(10), use_container_width=True)

        required_cols = {"pressure","temperature","dewpoint","humidity","cloud","sunshine","winddirection","windspeed"}
        if not required_cols.issubset(df_upload.columns):
            st.error(f"Missing columns: {required_cols - set(df_upload.columns)}")
        else:
            if st.button("🚀 Run Batch Prediction"):
                records = df_upload[list(required_cols)].to_dict(orient="records")
                with st.spinner(f"Processing {len(records)} records…"):
                    try:
                        resp = call_predict_batch(records)
                        resp.raise_for_status()
                        batch_result = resp.json()

                        preds = pd.DataFrame(batch_result["predictions"])
                        df_out = df_upload.copy()
                        df_out["prediction"]        = preds["final_prediction"].values
                        df_out["rain_probability"]   = preds["rain_probability"].values
                        df_out["confidence"]         = preds["confidence_level"].values

                        st.success(f"✅ {batch_result['total']} predictions complete!")

                        # Summary stats
                        rain_count    = (df_out["prediction"] == "Rain").sum()
                        no_rain_count = len(df_out) - rain_count
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="metric-card"><div class="val">{rain_count}</div><div class="lbl">🌧️ Rain Days</div></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="metric-card"><div class="val">{no_rain_count}</div><div class="lbl">☀️ No Rain Days</div></div>', unsafe_allow_html=True)
                        with col3:
                            avg_prob = round(df_out["rain_probability"].mean(), 1)
                            st.markdown(f'<div class="metric-card"><div class="val">{avg_prob}%</div><div class="lbl">📊 Avg Rain Prob</div></div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.dataframe(df_out.style.applymap(
                            lambda v: "background-color:#e3f2fd; color:#1565c0; font-weight:600" if v == "Rain"
                            else ("background-color:#e8f5e9; color:#2e7d32; font-weight:600" if v == "No Rain" else ""),
                            subset=["prediction"]
                        ), use_container_width=True)

                        # Pie chart
                        fig_pie = px.pie(
                            values=[rain_count, no_rain_count],
                            names=["Rain", "No Rain"],
                            color_discrete_sequence=["#1565c0", "#2e7d32"],
                            hole=0.4,
                            title="Prediction Distribution",
                        )
                        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320)
                        st.plotly_chart(fig_pie, use_container_width=True)

                        # Download results
                        csv_out = df_out.to_csv(index=False).encode()
                        st.download_button("⬇️ Download Results", csv_out, "batch_predictions.csv", "text/csv")

                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

# ══════════════════════════ TAB 3: DATA EXPLORER ══════════════════════════════
with tab_explore:
    st.markdown('<div class="section-title">📊 Dataset Explorer</div>', unsafe_allow_html=True)
    explore_file = st.file_uploader("Upload your Rainfall.csv to explore", type=["csv"], key="explorer")

    if explore_file:
        df_exp = pd.read_csv(explore_file)
        df_exp.columns = df_exp.columns.str.strip()

        st.markdown(f"**Shape:** {df_exp.shape[0]} rows × {df_exp.shape[1]} columns")
        st.dataframe(df_exp.head(20), use_container_width=True)

        st.markdown('<div class="section-title">📈 Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df_exp.describe().T.round(3), use_container_width=True)

        # Distribution plots
        st.markdown('<div class="section-title">📉 Feature Distributions</div>', unsafe_allow_html=True)
        num_cols = df_exp.select_dtypes(include=np.number).columns.tolist()
        selected = st.multiselect("Select features to plot", num_cols, default=num_cols[:4])
        if selected:
            cols_plot = st.columns(min(len(selected), 2))
            for i, feat in enumerate(selected):
                with cols_plot[i % 2]:
                    fig_hist = px.histogram(
                        df_exp, x=feat, nbins=30,
                        title=f"Distribution: {feat}",
                        color_discrete_sequence=["#1565c0"],
                    )
                    fig_hist.update_layout(
                        height=280, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

        # Correlation heatmap
        st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = df_exp[num_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Blues",
            text=corr.round(2).values,
            texttemplate="%{text}",
        ))
        fig_corr.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.info("Upload your `Rainfall.csv` file to explore the dataset.")

# ══════════════════════════ TAB 4: MODEL INFO ═════════════════════════════════
with tab_info:
    st.markdown('<div class="section-title">🤖 Ensemble Architecture</div>', unsafe_allow_html=True)
    info = get_model_info()

    if info:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            | Property | Value |
            |---|---|
            | **Ensemble Size** | {info['ensemble_size']} models |
            | **Voting Strategy** | Soft Voting (averaged probabilities) |
            | **Scaler** | {info['scaler_type']} |
            | **Input Features** | {info['ensemble_size'] and len(info['feature_names'])} |
            | **Output Classes** | {', '.join(info['prediction_classes'])} |
            """)
        with col2:
            st.markdown("**Models in Ensemble:**")
            icons = {"Logistic Regression": "📐", "XGBoost": "🚀", "SVC": "🎯"}
            for m in info["models"]:
                icon = icons.get(m, "🤖")
                st.markdown(f"- {icon} **{m}**")
    else:
        st.warning("Could not fetch model info from API.")

    st.markdown('<div class="section-title">⚙️ Input Features</div>', unsafe_allow_html=True)
    features_meta = [
        {"Feature": "pressure",      "Description": "Atmospheric pressure",   "Unit": "hPa",   "Range": "900–1100"},
        {"Feature": "temperature",   "Description": "Mean temperature",        "Unit": "°C",    "Range": "-20–60"},
        {"Feature": "dewpoint",      "Description": "Dew point temperature",   "Unit": "°C",    "Range": "-30–40"},
        {"Feature": "humidity",      "Description": "Relative humidity",       "Unit": "%",     "Range": "0–100"},
        {"Feature": "cloud",         "Description": "Cloud cover",             "Unit": "oktas", "Range": "0–8"},
        {"Feature": "sunshine",      "Description": "Sunshine hours per day",  "Unit": "hrs",   "Range": "0–24"},
        {"Feature": "winddirection", "Description": "Wind direction",          "Unit": "°",     "Range": "0–360"},
        {"Feature": "windspeed",     "Description": "Wind speed",              "Unit": "km/h",  "Range": "0–200"},
    ]
    st.dataframe(pd.DataFrame(features_meta), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">🚀 API Endpoints</div>', unsafe_allow_html=True)
    st.markdown("""
    | Method | Endpoint | Description |
    |---|---|---|
    | `GET` | `/` | Root / Welcome |
    | `GET` | `/health` | Health check & model status |
    | `GET` | `/model-info` | Ensemble details |
    | `GET` | `/features` | Feature definitions |
    | `POST` | `/predict` | Single prediction |
    | `POST` | `/predict/batch` | Batch prediction (up to 100 records) |
    | `GET` | `/docs` | Interactive Swagger UI |
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<footer class="raincast-footer">
    🌧️ RainCast AI &nbsp;•&nbsp; Ensemble ML Model &nbsp;•&nbsp; 
    Logistic Regression + XGBoost + SVC &nbsp;•&nbsp;
    Built with FastAPI & Streamlit
</footer>
""", unsafe_allow_html=True)
