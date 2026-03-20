"""
🏍️ Bike Price Prediction App
================================
Run with:  streamlit run bike_price_app.py
Install :  pip install -r requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ───────────────────────── Page Config ─────────────────────────────
st.set_page_config(
    page_title="🏍️ Bike Price Predictor",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────── Custom CSS ──────────────────────────────
st.markdown("""
<style>
    .main { background: #0a0a14; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #12122a 0%, #1a1a35 100%);
        border: 1px solid #ff6b35;
        border-radius: 14px;
        padding: 18px 22px;
    }
    [data-testid="stMetricLabel"] { color: #a0aec0 !important; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #fff !important; font-size: 26px; font-weight: 800; }
    [data-testid="stMetricDelta"] { color: #68d391 !important; }

    /* Prediction banner */
    .pred-banner {
        background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 40%, #ff6b35 100%);
        border-radius: 18px;
        padding: 30px 36px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 10px 40px rgba(255,107,53,0.45);
    }
    .pred-banner h1 { color: #0a0a14; font-size: 3.5rem; margin: 0; font-weight: 900; }
    .pred-banner p  { color: rgba(10,10,20,0.75); font-size: 1rem; margin: 4px 0 0; font-weight: 600; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #ff6b35, #f7c59f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        font-weight: 800;
        margin: 24px 0 8px;
        letter-spacing: 0.5px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1f 0%, #12122a 100%) !important;
        border-right: 2px solid #ff6b35;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #12122a;
        border-radius: 8px;
        color: #a0aec0;
        padding: 8px 18px;
        border: 1px solid #1a1a35;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b35, #f7c59f) !important;
        color: #0a0a14 !important;
        font-weight: 700;
    }

    /* Insight boxes */
    .insight-box {
        background: #12122a;
        border-left: 4px solid #ff6b35;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin: 6px 0;
        color: #e2e8f0;
        font-size: 0.88rem;
    }

    /* Range badge */
    .range-badge {
        display: inline-block;
        background: #12122a;
        border: 1px solid #ff6b35;
        border-radius: 8px;
        padding: 6px 14px;
        color: #f7c59f;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Plotly theme ────────────────────────────
DARK_BG  = "#0a0a14"
CARD_BG  = "#12122a"
FONT_CLR = "#e2e8f0"
ORANGE   = "#ff6b35"
PEACH    = "#f7c59f"
PALETTE  = [ORANGE, PEACH, "#ff9f7f", "#ffcba4", "#e05a22",
            "#ffd4b8", "#c44b1c", "#ffebd9"]

def dark_layout(fig, height=400):
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=DARK_BG,
        font_color=FONT_CLR, height=height,
        margin=dict(l=16, r=16, t=40, b=16),
    )
    return fig

# ───────────────────────── Data Loading ────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("bike_price_data.csv")
    df.dropna(inplace=True)
    df["Car Age"] = 2024 - df["Year"]
    return df

@st.cache_resource
def train_models(df):
    FEATURES = [
        "Year", "Engine CC", "Power (BHP)", "Mileage (kmpl)",
        "Kms Driven", "Car Age",
        "Brand", "Fuel Type", "Bike Type", "Condition", "Transmission", "Owner Type"
    ]
    TARGET = "Price"

    X = df[FEATURES].copy()
    y = df[TARGET]

    CAT_COLS = ["Brand", "Fuel Type", "Bike Type", "Condition", "Transmission", "Owner Type"]
    le_map = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.07,
            max_depth=5, subsample=0.85, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=14, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=14, random_state=42
        ),
        "Ridge Regression": Ridge(alpha=50.0),
    }

    results = {}
    scaler_ridge = StandardScaler()
    Xtr_s = scaler_ridge.fit_transform(X_train)
    Xte_s = scaler_ridge.transform(X_test)

    for name, model in models.items():
        Xtr = Xtr_s if name == "Ridge Regression" else X_train
        Xte = Xte_s if name == "Ridge Regression" else X_test

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        results[name] = {
            "model":  model,
            "scaler": scaler_ridge if name == "Ridge Regression" else None,
            "preds":  preds,
            "MAE":    mean_absolute_error(y_test, preds),
            "RMSE":   np.sqrt(mean_squared_error(y_test, preds)),
            "R²":     r2_score(y_test, preds),
            "MAPE":   np.mean(np.abs((y_test - preds) / y_test)) * 100,
            "y_test": y_test,
        }

    gb    = results["Gradient Boosting"]["model"]
    imps  = pd.Series(gb.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return results, le_map, FEATURES, imps, X_train, y_train, scaler_ridge

# ───────────────────────── Load ────────────────────────────────────
df = load_data()
results, le_map, FEATURES, importances, X_train, y_train, scaler_ridge = train_models(df)

# ═══════════════════════ SIDEBAR ═══════════════════════════════════
with st.sidebar:
    st.markdown("## 🏍️ Bike Price Predictor")
    st.markdown("---")
    st.markdown("#### 🔧 Enter Bike Details")

    brand        = st.selectbox("Brand",          sorted(df["Brand"].unique()))
    fuel_type    = st.selectbox("Fuel Type",       sorted(df["Fuel Type"].unique()))
    bike_type    = st.selectbox("Bike Type",       sorted(df["Bike Type"].unique()))
    condition    = st.selectbox("Condition",       sorted(df["Condition"].unique()))
    transmission = st.selectbox("Transmission",    sorted(df["Transmission"].unique()))
    owner_type   = st.selectbox("Owner Type",      sorted(df["Owner Type"].unique()))

    st.markdown("---")
    year         = st.slider("Year",              2005, 2024, 2018)
    engine_cc    = st.slider("Engine CC",           100, 1340, 350, 10)
    power_bhp    = st.slider("Power (BHP)",          5.0, 200.0, 35.0, 0.5)
    mileage_kmpl = st.slider("Mileage (kmpl)",      15.0, 90.0, 40.0, 0.5)
    kms_driven   = st.slider("Kms Driven",            0, 120_000, 20_000, 500)

    st.markdown("---")
    model_choice = st.selectbox("🤖 ML Model", list(results.keys()))

    predict_btn  = st.button("🔮 Predict Price", use_container_width=True, type="primary")

# ───────────────────────── Prediction ──────────────────────────────
def make_prediction(brand, fuel_type, bike_type, condition, transmission,
                    owner_type, year, engine_cc, power_bhp,
                    mileage_kmpl, kms_driven, model_choice):
    car_age = 2024 - year
    row = pd.DataFrame([{
        "Year": year, "Engine CC": engine_cc, "Power (BHP)": power_bhp,
        "Mileage (kmpl)": mileage_kmpl, "Kms Driven": kms_driven,
        "Car Age": car_age,
        "Brand": brand, "Fuel Type": fuel_type, "Bike Type": bike_type,
        "Condition": condition, "Transmission": transmission, "Owner Type": owner_type,
    }])
    for col in ["Brand","Fuel Type","Bike Type","Condition","Transmission","Owner Type"]:
        row[col] = le_map[col].transform(row[col].astype(str))

    info = results[model_choice]
    X_in = info["scaler"].transform(row) if info["scaler"] else row
    return max(0, info["model"].predict(X_in)[0])

if predict_btn:
    pred = make_prediction(
        brand, fuel_type, bike_type, condition, transmission,
        owner_type, year, engine_cc, power_bhp,
        mileage_kmpl, kms_driven, model_choice
    )
    st.session_state.update({"pred": pred, "mod": model_choice, "brd": brand, "btp": bike_type})

# ═══════════════════════ MAIN AREA ═════════════════════════════════
st.markdown("# 🏍️ Bike Price Prediction Dashboard")
st.markdown("*Predict resale value · Explore trends · Compare ML models*")

if "pred" in st.session_state:
    p = st.session_state["pred"]
    m = st.session_state["mod"]
    b = st.session_state["brd"]
    t = st.session_state["btp"]

    st.markdown(f"""
    <div class="pred-banner">
        <p>⚡ Predicted Price  ·  Model: <b>{m}</b></p>
        <h1>₹ {p:,.0f}</h1>
        <p>Estimated market value · {b} · {t}</p>
    </div>
    """, unsafe_allow_html=True)

    mae = results[m]["MAE"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conservative Est.", f"₹{max(0,p-mae):,.0f}", "lower bound")
    c2.metric("Predicted Price",   f"₹{p:,.0f}",            "estimate")
    c3.metric("Optimistic Est.",   f"₹{p+mae:,.0f}",        "upper bound")
    c4.metric("Model R² Score",    f"{results[m]['R²']:.3f}", "accuracy")
    st.markdown("---")

# ───────────────────────── Tabs ────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔍 EDA", "🤖 Model Performance",
    "📈 Feature Importance", "🔗 Correlations", "🏷️ Brand Analysis"
])

# ══════════ TAB 1 — OVERVIEW ══════════
with tab1:
    st.markdown('<p class="section-header">📦 Dataset Snapshot</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Bikes",    f"{len(df):,}")
    c2.metric("Avg Price",      f"₹{df['Price'].mean():,.0f}")
    c3.metric("Median Price",   f"₹{df['Price'].median():,.0f}")
    c4.metric("Brands",         df["Brand"].nunique())
    c5.metric("Avg Engine CC",  f"{df['Engine CC'].mean():,.0f} cc")

    # Box plot – price by brand
    st.markdown('<p class="section-header">Price Distribution by Brand</p>', unsafe_allow_html=True)
    fig = px.box(
        df, x="Brand", y="Price", color="Brand",
        color_discrete_sequence=PALETTE * 2, template="plotly_dark",
        category_orders={"Brand": df.groupby("Brand")["Price"].median().sort_values(ascending=False).index.tolist()}
    )
    dark_layout(fig, 440).update_layout(showlegend=False, xaxis_title="", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="section-header">Bike Type Split</p>', unsafe_allow_html=True)
        bt = df["Bike Type"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=bt.index, values=bt.values, hole=0.55,
            marker_colors=PALETTE
        ))
        dark_layout(fig2, 300)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Condition Split</p>', unsafe_allow_html=True)
        cond = df["Condition"].value_counts()
        fig3 = go.Figure(go.Bar(
            x=cond.index, y=cond.values,
            marker_color=[ORANGE, PEACH, "#e05a22"],
            text=cond.values, textposition="auto"
        ))
        dark_layout(fig3, 300).update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)

    with col3:
        st.markdown('<p class="section-header">Fuel Type</p>', unsafe_allow_html=True)
        fuel = df["Fuel Type"].value_counts()
        fig4 = go.Figure(go.Pie(
            labels=fuel.index, values=fuel.values, hole=0.55,
            marker_colors=[ORANGE, PEACH]
        ))
        dark_layout(fig4, 300)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<p class="section-header">Sample Data</p>', unsafe_allow_html=True)
    st.dataframe(
        df.drop(columns=["Car Age"]).sample(10, random_state=7).reset_index(drop=True),
        use_container_width=True
    )

# ══════════ TAB 2 — EDA ══════════
with tab2:
    st.markdown('<p class="section-header">Price vs Kms Driven</p>', unsafe_allow_html=True)
    fig = px.scatter(
        df, x="Kms Driven", y="Price", color="Bike Type",
        size="Engine CC", hover_data=["Brand","Year","Condition"],
        color_discrete_sequence=PALETTE, template="plotly_dark", opacity=0.65
    )
    dark_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Engine CC vs Price</p>', unsafe_allow_html=True)
        fig5 = px.scatter(
            df, x="Engine CC", y="Price", color="Condition",
            color_discrete_sequence=[ORANGE, PEACH, "#e05a22"],
            template="plotly_dark", opacity=0.6
        )
        dark_layout(fig5, 340)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Power (BHP) vs Price</p>', unsafe_allow_html=True)
        fig6 = px.scatter(
            df, x="Power (BHP)", y="Price", color="Fuel Type",
            color_discrete_sequence=[ORANGE, PEACH],
            template="plotly_dark", opacity=0.6
        )
        dark_layout(fig6, 340)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<p class="section-header">Price by Year (Trend)</p>', unsafe_allow_html=True)
    yr_avg = df.groupby("Year")["Price"].mean().reset_index()
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=yr_avg["Year"], y=yr_avg["Price"],
        mode="lines+markers",
        line=dict(color=ORANGE, width=3),
        marker=dict(size=7, color=PEACH),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.12)"
    ))
    dark_layout(fig7, 340).update_layout(xaxis_title="Year", yaxis_title="Avg Price (₹)")
    st.plotly_chart(fig7, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Mileage vs Price</p>', unsafe_allow_html=True)
        fig8 = px.scatter(
            df, x="Mileage (kmpl)", y="Price", color="Bike Type",
            color_discrete_sequence=PALETTE, template="plotly_dark", opacity=0.6
        )
        dark_layout(fig8, 330)
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Price Distribution (Histogram)</p>', unsafe_allow_html=True)
        fig9 = px.histogram(
            df, x="Price", nbins=70, color="Condition",
            color_discrete_sequence=[ORANGE, PEACH, "#e05a22"],
            template="plotly_dark", barmode="overlay", opacity=0.75
        )
        dark_layout(fig9, 330)
        st.plotly_chart(fig9, use_container_width=True)

    # Seaborn heatmap — Avg price brand × bike type
    st.markdown('<p class="section-header">Brand × Bike Type — Avg Price Heatmap</p>', unsafe_allow_html=True)
    pivot = df.pivot_table(index="Brand", columns="Bike Type", values="Price", aggfunc="mean")
    fig_h, ax = plt.subplots(figsize=(13, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    sns.heatmap(
        pivot / 1000, annot=True, fmt=".0f",
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        linewidths=0.4, linecolor=DARK_BG, ax=ax,
        annot_kws={"size": 9, "color": "white"},
        cbar_kws={"label": "Avg Price (₹ 000s)"}
    )
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.set_title("Average Price (₹ 000s) — Brand × Bike Type", color="white", pad=10)
    plt.xticks(color="white", rotation=30); plt.yticks(color="white", rotation=0)
    plt.tight_layout()
    st.pyplot(fig_h)

# ══════════ TAB 3 — MODEL PERFORMANCE ══════════
with tab3:
    st.markdown('<p class="section-header">Model Comparison Table</p>', unsafe_allow_html=True)

    rows_ = []
    for name, info in results.items():
        rows_.append({
            "Model":     name,
            "R² Score":  round(info["R²"],   4),
            "MAE (₹)":   f"₹{info['MAE']:,.0f}",
            "RMSE (₹)":  f"₹{info['RMSE']:,.0f}",
            "MAPE (%)":  round(info["MAPE"],  2),
        })
    st.dataframe(pd.DataFrame(rows_).set_index("Model"), use_container_width=True)

    # Metric bar chart
    names_  = list(results.keys())
    r2s_    = [results[n]["R²"]   for n in names_]
    maes_   = [results[n]["MAE"]  / 1000 for n in names_]
    rmses_  = [results[n]["RMSE"] / 1000 for n in names_]
    mapes_  = [results[n]["MAPE"] for n in names_]

    fig_m = make_subplots(rows=1, cols=4,
        subplot_titles=["R² (↑ better)", "MAE ₹000 (↓ better)",
                        "RMSE ₹000 (↓ better)", "MAPE % (↓ better)"])
    for i, vals in enumerate([r2s_, maes_, rmses_, mapes_], 1):
        fig_m.add_trace(
            go.Bar(x=names_, y=vals,
                   marker_color=PALETTE[:4],
                   text=[f"{v:.2f}" for v in vals],
                   textposition="auto"),
            row=1, col=i
        )
    fig_m.update_layout(
        template="plotly_dark", showlegend=False, height=370,
        plot_bgcolor=CARD_BG, paper_bgcolor=DARK_BG, font_color=FONT_CLR
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # Actual vs Predicted
    st.markdown(f'<p class="section-header">Actual vs Predicted — {model_choice}</p>', unsafe_allow_html=True)
    info      = results[model_choice]
    y_test_v  = info["y_test"].values
    y_pred_v  = info["preds"]

    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=y_test_v, y=y_pred_v, mode="markers",
        marker=dict(color=ORANGE, opacity=0.45, size=5), name="Predicted"
    ))
    lo, hi = y_test_v.min(), y_test_v.max()
    fig_ap.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=PEACH, dash="dash", width=2), name="Perfect fit"
    ))
    dark_layout(fig_ap, 430).update_layout(
        xaxis_title="Actual Price (₹)", yaxis_title="Predicted Price (₹)"
    )
    st.plotly_chart(fig_ap, use_container_width=True)

    # Residuals
    st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
    residuals = y_test_v - y_pred_v
    col1, col2 = st.columns(2)
    with col1:
        fig_r = px.histogram(
            x=residuals, nbins=70,
            color_discrete_sequence=[ORANGE],
            template="plotly_dark", labels={"x": "Residual (₹)"}
        )
        fig_r.add_vline(x=0, line_color=PEACH, line_dash="dash", line_width=2)
        dark_layout(fig_r, 320).update_layout(xaxis_title="Residual (₹)", yaxis_title="Count")
        st.plotly_chart(fig_r, use_container_width=True)
    with col2:
        # Residual % distribution
        residual_pct = residuals / y_test_v * 100
        fig_rp = px.histogram(
            x=residual_pct, nbins=70,
            color_discrete_sequence=[PEACH],
            template="plotly_dark", labels={"x": "Residual %"}
        )
        fig_rp.add_vline(x=0, line_color=ORANGE, line_dash="dash", line_width=2)
        dark_layout(fig_rp, 320).update_layout(xaxis_title="Residual %", yaxis_title="Count")
        st.plotly_chart(fig_rp, use_container_width=True)

    # Error stats
    st.markdown('<p class="section-header">Error Statistics</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean Residual",  f"₹{residuals.mean():,.0f}")
    c2.metric("Std of Residual",f"₹{residuals.std():,.0f}")
    c3.metric("Within ±10%",    f"{(np.abs(residual_pct)<=10).mean()*100:.1f}%")
    c4.metric("Within ±20%",    f"{(np.abs(residual_pct)<=20).mean()*100:.1f}%")

# ══════════ TAB 4 — FEATURE IMPORTANCE ══════════
with tab4:
    st.markdown('<p class="section-header">Feature Importance — Gradient Boosting</p>', unsafe_allow_html=True)

    imp_df = importances.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    imp_df["Pct"] = imp_df["Importance"] / imp_df["Importance"].sum() * 100

    fig_fi = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[CARD_BG, ORANGE, PEACH],
        template="plotly_dark",
        text=imp_df["Pct"].apply(lambda x: f"{x:.1f}%")
    )
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=DARK_BG, font_color=FONT_CLR,
        height=500, yaxis=dict(autorange="reversed"), coloraxis_showscale=False
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Insights
    st.markdown('<p class="section-header">Key Insights</p>', unsafe_allow_html=True)
    for _, row in imp_df.iterrows():
        st.markdown(
            f'<div class="insight-box">🔸 <b>{row["Feature"]}</b> — '
            f'contributes <b>{row["Pct"]:.1f}%</b> to predictions</div>',
            unsafe_allow_html=True
        )

    # Partial dependence — price vs engine cc
    st.markdown('<p class="section-header">Sensitivity: Engine CC → Price</p>', unsafe_allow_html=True)
    cc_range = np.linspace(df["Engine CC"].min(), df["Engine CC"].max(), 200)
    base_enc = {
        "Year": 2018, "Engine CC": 0, "Power (BHP)": 35, "Mileage (kmpl)": 40,
        "Kms Driven": 20000, "Car Age": 6,
        "Brand":        le_map["Brand"].transform(["Honda"])[0],
        "Fuel Type":    le_map["Fuel Type"].transform(["Petrol"])[0],
        "Bike Type":    le_map["Bike Type"].transform(["Commuter"])[0],
        "Condition":    le_map["Condition"].transform(["Used"])[0],
        "Transmission": le_map["Transmission"].transform(["Manual"])[0],
        "Owner Type":   le_map["Owner Type"].transform(["1st Owner"])[0],
    }
    base_rows = pd.DataFrame([base_enc] * 200)
    base_rows["Engine CC"] = cc_range
    pdp_preds = results["Gradient Boosting"]["model"].predict(base_rows)

    fig_pdp = go.Figure()
    fig_pdp.add_trace(go.Scatter(
        x=cc_range, y=pdp_preds, mode="lines",
        line=dict(color=ORANGE, width=3),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.12)"
    ))
    dark_layout(fig_pdp, 320).update_layout(
        xaxis_title="Engine CC", yaxis_title="Predicted Price (₹)"
    )
    st.plotly_chart(fig_pdp, use_container_width=True)

    # Partial dependence — price vs kms driven
    st.markdown('<p class="section-header">Sensitivity: Kms Driven → Price</p>', unsafe_allow_html=True)
    kms_range = np.linspace(0, 120000, 200)
    base_rows2 = pd.DataFrame([base_enc] * 200)
    base_rows2["Engine CC"]  = 350
    base_rows2["Kms Driven"] = kms_range
    pdp2 = results["Gradient Boosting"]["model"].predict(base_rows2)

    fig_pdp2 = go.Figure()
    fig_pdp2.add_trace(go.Scatter(
        x=kms_range, y=pdp2, mode="lines",
        line=dict(color=PEACH, width=3),
        fill="tozeroy", fillcolor="rgba(247,197,159,0.12)"
    ))
    dark_layout(fig_pdp2, 300).update_layout(
        xaxis_title="Kms Driven", yaxis_title="Predicted Price (₹)"
    )
    st.plotly_chart(fig_pdp2, use_container_width=True)

# ══════════ TAB 5 — CORRELATIONS ══════════
with tab5:
    st.markdown('<p class="section-header">Numeric Correlation Matrix</p>', unsafe_allow_html=True)
    num_cols = ["Year","Engine CC","Power (BHP)","Mileage (kmpl)","Kms Driven","Car Age","Price"]
    corr = df[num_cols].corr()

    fig_c, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="YlOrRd", center=0,
        linewidths=0.4, linecolor=DARK_BG, ax=ax,
        annot_kws={"size": 10, "color": "black"}
    )
    ax.tick_params(colors="white")
    ax.set_title("Correlation Matrix", color="white", pad=10)
    plt.xticks(color="white", rotation=30); plt.yticks(color="white", rotation=0)
    plt.tight_layout()
    st.pyplot(fig_c)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Avg Price by Transmission</p>', unsafe_allow_html=True)
        t_avg = df.groupby("Transmission")["Price"].mean().reset_index().sort_values("Price", ascending=False)
        fig_t = px.bar(
            t_avg, x="Transmission", y="Price",
            color="Transmission", color_discrete_sequence=PALETTE,
            template="plotly_dark", text="Price"
        )
        fig_t.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        dark_layout(fig_t, 320).update_layout(showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Avg Price by Owner Type</p>', unsafe_allow_html=True)
        o_avg = df.groupby("Owner Type")["Price"].mean().reset_index().sort_values("Price", ascending=False)
        fig_o = px.bar(
            o_avg, x="Owner Type", y="Price",
            color="Owner Type", color_discrete_sequence=PALETTE,
            template="plotly_dark", text="Price"
        )
        fig_o.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        dark_layout(fig_o, 320).update_layout(showlegend=False)
        st.plotly_chart(fig_o, use_container_width=True)

    # Seaborn pairplot (static)
    st.markdown('<p class="section-header">Pairplot — Key Numerics</p>', unsafe_allow_html=True)
    pair_cols = ["Engine CC", "Power (BHP)", "Kms Driven", "Price"]
    sample_df = df[pair_cols + ["Condition"]].sample(500, random_state=1)
    fig_pp = sns.pairplot(
        sample_df, hue="Condition",
        palette={c: col for c, col in zip(df["Condition"].unique(), [ORANGE, PEACH, "#e05a22"])},
        plot_kws={"alpha": 0.5, "s": 18},
        diag_kws={"fill": True}
    )
    fig_pp.figure.patch.set_facecolor(DARK_BG)
    for ax_row in fig_pp.axes:
        for ax_ in ax_row:
            ax_.set_facecolor(CARD_BG)
            ax_.tick_params(colors="white")
            ax_.xaxis.label.set_color("white")
            ax_.yaxis.label.set_color("white")
    st.pyplot(fig_pp.figure)

# ══════════ TAB 6 — BRAND ANALYSIS ══════════
with tab6:
    st.markdown('<p class="section-header">Brand Deep Dive</p>', unsafe_allow_html=True)

    selected_brand = st.selectbox("Select Brand to Analyse", sorted(df["Brand"].unique()), key="ba")
    bdf = df[df["Brand"] == selected_brand]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models",      bdf["Model"].nunique())
    c2.metric("Avg Price",   f"₹{bdf['Price'].mean():,.0f}")
    c3.metric("Avg Engine",  f"{bdf['Engine CC'].mean():,.0f} cc")
    c4.metric("Avg BHP",     f"{bdf['Power (BHP)'].mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Price by Model</p>', unsafe_allow_html=True)
        fig_bm = px.box(
            bdf, x="Model", y="Price", color="Model",
            color_discrete_sequence=PALETTE * 2, template="plotly_dark"
        )
        dark_layout(fig_bm, 350).update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(fig_bm, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Price by Condition</p>', unsafe_allow_html=True)
        fig_bc = px.violin(
            bdf, x="Condition", y="Price", color="Condition",
            color_discrete_sequence=[ORANGE, PEACH, "#e05a22"],
            template="plotly_dark", box=True, points=False
        )
        dark_layout(fig_bc, 350).update_layout(showlegend=False)
        st.plotly_chart(fig_bc, use_container_width=True)

    st.markdown('<p class="section-header">Year-wise Avg Price Trend</p>', unsafe_allow_html=True)
    yr_b = bdf.groupby("Year")["Price"].mean().reset_index()
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=yr_b["Year"], y=yr_b["Price"],
        mode="lines+markers",
        line=dict(color=ORANGE, width=3),
        marker=dict(size=8, color=PEACH),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.1)"
    ))
    dark_layout(fig_bt, 320).update_layout(
        xaxis_title="Year", yaxis_title="Avg Price (₹)",
        title=dict(text=f"{selected_brand} — Avg Price by Year", font_color="white")
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Top 10 most listed models
    st.markdown('<p class="section-header">Most Listed Models</p>', unsafe_allow_html=True)
    top_models = bdf["Model"].value_counts().head(10)
    fig_tm = px.bar(
        x=top_models.index, y=top_models.values,
        color=top_models.values,
        color_continuous_scale=[CARD_BG, ORANGE, PEACH],
        template="plotly_dark",
        text=top_models.values,
        labels={"x": "Model", "y": "Count"}
    )
    fig_tm.update_traces(textposition="outside")
    dark_layout(fig_tm, 320).update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_tm, use_container_width=True)

# ───────────────────────── Footer ──────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:0.78rem;'>"
    "🏍️ Bike Price Predictor · Built with Streamlit · scikit-learn · Pandas · "
    "Seaborn · Matplotlib · Plotly"
    "</p>",
    unsafe_allow_html=True
)
