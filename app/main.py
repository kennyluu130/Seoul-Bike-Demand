import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Seoul Bike Demand · Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  THEME / CUSTOM CSS
# ─────────────────────────────────────────────
BLUE   = "#003478"
RED    = "#C60C30"
LIGHT  = "#F4F6FB"
WHITE  = "#FFFFFF"
GREY   = "#6B7280"
ACCENT = "#FF6B35"

st.markdown(f"""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
  }}

  /* ── Top bar accent ── */
  [data-testid="stHeader"] {{
    background: {BLUE};
  }}

  /* ── Main background ── */
  .main .block-container {{
    background: {LIGHT};
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
  }}

  /* ── Hero banner ── */
  .hero {{
    background: linear-gradient(135deg, {BLUE} 0%, #004fa3 60%, {RED} 140%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    color: white;
    margin-bottom: 2rem;
  }}
  .hero h1 {{
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }}
  .hero p {{
    font-size: 1.05rem;
    opacity: 0.85;
    margin: 0;
    font-weight: 300;
  }}

  /* ── KPI cards ── */
  .kpi-row {{
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
  }}
  .kpi-card {{
    flex: 1;
    min-width: 140px;
    background: {WHITE};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 4px solid {BLUE};
  }}
  .kpi-card.red {{ border-left-color: {RED}; }}
  .kpi-card.orange {{ border-left-color: {ACCENT}; }}
  .kpi-label {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {GREY};
    margin-bottom: 0.3rem;
  }}
  .kpi-value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {BLUE};
    font-family: 'IBM Plex Mono', monospace;
  }}
  .kpi-card.red .kpi-value {{ color: {RED}; }}
  .kpi-card.orange .kpi-value {{ color: {ACCENT}; }}
  .kpi-sub {{
    font-size: 0.78rem;
    color: {GREY};
    margin-top: 0.2rem;
  }}

  /* ── Section cards ── */
  .section-card {{
    background: {WHITE};
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
  }}
  .section-title {{
    font-size: 1rem;
    font-weight: 700;
    color: {BLUE};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}

  /* ── Tabs ── */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0.25rem;
    background: {WHITE};
    padding: 0.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }}
  [data-testid="stTabs"] [data-baseweb="tab"] {{
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.9rem;
  }}
  [data-testid="stTabs"] [aria-selected="true"] {{
    background: {BLUE} !important;
    color: white !important;
  }}

  /* ── Metric overrides ── */
  [data-testid="metric-container"] {{
    background: {WHITE};
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}

  /* ── Prediction badge ── */
  .pred-badge {{
    display: inline-block;
    padding: 0.6rem 1.4rem;
    border-radius: 999px;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.5rem;
  }}
  .pred-low    {{ background: #e8f5e9; color: #2e7d32; }}
  .pred-mid    {{ background: #fff3e0; color: #e65100; }}
  .pred-high   {{ background: #fce4ec; color: #c62828; }}

  /* ── Leaderboard table ── */
  .leaderboard th {{
    background: {BLUE};
    color: white;
    font-weight: 600;
    padding: 0.6rem 1rem;
  }}
  .leaderboard td {{
    padding: 0.55rem 1rem;
    border-bottom: 1px solid #e5e7eb;
  }}
  .leaderboard tr:first-child td {{ font-weight: 700; color: {BLUE}; }}

  /* ── About tag ── */
  .tag {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.2rem 0.15rem;
    background: {LIGHT};
    color: {BLUE};
    border: 1px solid #d1d5db;
  }}

  div[data-testid="stForm"] {{
    background: {WHITE};
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH  = BASE_DIR / "data" / "SeoulBikeData.csv"

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FEATURE_ORDER = [
    "Hour", "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)",
    "Dew point temperature(°C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)",
    "Snowfall (cm)", "Holiday", "Functioning Day", "Month"
]

# Defaults = realistic Seoul summer weekday evening peak
# (temp/dew/humidity must be internally consistent or tree models pull predictions low)
FEATURES_INFO = {
    "Hour":                        (0,    23,    18),    # 18:00 = evening rush, highest demand
    "Month":                       (1,    12,    6),     # June = peak season
    "Holiday":                     (0,    1,     0),     # working day
    "Functioning Day":             (0,    1,     1),     # operating
    "Temperature(°C)":             (-17.8, 39.4, 22.0), # warm summer evening
    "Dew point temperature(°C)":   (-30.6, 27.2, 15.0), # consistent with ~60% humidity at 22°C
    "Humidity(%)":                 (0.0,  98.0,  60.0), # typical Seoul summer
    "Wind speed (m/s)":            (0.0,  7.4,   1.5),  # light breeze
    "Visibility (10m)":            (27.0, 2000.0, 1600.0), # clear evening
    "Solar Radiation (MJ/m2)":     (0.0,  3.52,  0.5),  # late afternoon declining
    "Rainfall(mm)":                (0.0,  35.0,  0.0),  # dry
    "Snowfall (cm)":               (0.0,  8.8,   0.0),  # no snow
}

NUMERIC_FEATURES = [
    "Temperature(°C)", "Dew point temperature(°C)", "Humidity(%)",
    "Wind speed (m/s)", "Visibility (10m)", "Solar Radiation (MJ/m2)",
    "Rainfall(mm)", "Snowfall (cm)"
]

MODEL_FILES = {
    "Random Forest": ("rf.pkl",   "joblib"),
    "XGBoost":       ("xgb.pkl",  "pickle"),
    "LightGBM":      ("lgbm.pkl", "pickle"),
    "KNN":           ("knn.pkl",  "pickle"),
    "Linear Regression": ("linear.pkl", "pickle"),
    "Ridge":         ("ridge.pkl","pickle"),
    "Lasso":         ("lasso.pkl","pickle"),
}

# Hand-entered from notebooks
MODEL_METRICS = {
    "Random Forest":     {"R²": 0.8517, "MSE": 61794,  "MAE": 141},
    "XGBoost":           {"R²": 0.8771, "MSE": 51221,  "MAE": 127}, 
    "LightGBM":          {"R²": 0.8707, "MSE": 53869,  "MAE": 131},
    "KNN":               {"R²": 0.4258, "MSE": 89353,  "MAE": 172},
    "Linear Regression": {"R²": 0.4230, "MSE": 240392, "MAE": 309},
    "Ridge":             {"R²": 0.4237, "MSE": 240103, "MAE": 309},
    "Lasso":             {"R²": 0.4258, "MSE": 239241, "MAE": 309},
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    loaded = {}
    for name, (file, method) in MODEL_FILES.items():
        path = MODELS_DIR / file
        if not path.exists():
            continue
        try:
            if method == "joblib":
                loaded[name] = joblib.load(path)
            else:
                with open(path, "rb") as f:
                    loaded[name] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    return loaded


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """Load the raw Seoul Bike dataset. Returns None if not found."""
    if not DATA_PATH.exists():
        # Try common alternative locations
        for alt in [
            BASE_DIR / "SeoulBikeData.csv",
            Path(__file__).parent / "SeoulBikeData.csv",
        ]:
            if alt.exists():
                return pd.read_csv(alt, encoding="unicode_escape")
        return None

    return pd.read_csv(DATA_PATH, encoding="unicode_escape")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing to align column names."""
    df = df.copy()
    rename = {
        "Rented Bike Count": "BikeCount",
        "Temperature(°C)": "Temperature(°C)",
        "Humidity(%)": "Humidity(%)",
        "Wind speed (m/s)": "Wind speed (m/s)",
        "Visibility (10m)": "Visibility (10m)",
        "Dew point temperature(°C)": "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)": "Solar Radiation (MJ/m2)",
        "Rainfall(mm)": "Rainfall(mm)",
        "Snowfall (cm)": "Snowfall (cm)",
        "Seasons": "Season",
        "Holiday": "Holiday",
        "Functioning Day": "Functioning Day",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df["Month"] = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0=Mon
        df["IsWeekend"] = df["DayOfWeek"] >= 5

    if "Holiday" in df.columns:
        df["Holiday"] = df["Holiday"].map({"Holiday": 1, "No Holiday": 0}).fillna(df["Holiday"])
    if "Functioning Day" in df.columns:
        df["Functioning Day"] = df["Functioning Day"].map({"Yes": 1, "No": 0}).fillna(df["Functioning Day"])

    return df

# ─────────────────────────────────────────────
#  MATPLOTLIB STYLE HELPERS
# ─────────────────────────────────────────────
def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", color=BLUE, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=GREY)
    ax.set_ylabel(ylabel, fontsize=9, color=GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#e5e7eb")
    ax.set_facecolor(WHITE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(GREY)


def fig_bg(fig):
    fig.patch.set_facecolor(WHITE)
    return fig


# ─────────────────────────────────────────────
#  PREDICT HELPER
# ─────────────────────────────────────────────
# Threshold below which we suspect the model was trained on log1p-transformed target.
_LOG_THRESHOLD = 15.0

def predict(model, inputs: dict, inverse_log: bool = False) -> tuple[float, bool]:
    """Returns (predicted_bike_count, log_transform_was_applied)."""
    input_df = pd.DataFrame([inputs]).reindex(columns=FEATURE_ORDER)
    try:
        raw = float(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0, False

    # Auto-detect: if functioning day = 1 and raw < threshold → log-space
    is_functioning = inputs.get("Functioning Day", 1) == 1
    auto_log = is_functioning and (raw < _LOG_THRESHOLD) and (raw > 0)

    if inverse_log or auto_log:
        return max(0.0, float(np.expm1(raw))), True
    return max(0.0, raw), False


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
models = load_models()
raw_df = load_data()
df     = preprocess(raw_df) if raw_df is not None else None

has_data = df is not None and "BikeCount" in df.columns

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚲 Seoul Bike Sharing Demand</h1>
  <p>Machine-learning dashboard · Regression analysis · Demand forecasting</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI ROW  (data-driven if dataset available)
# ─────────────────────────────────────────────
if has_data:
    func_df    = df[(df["Functioning Day"].astype(str).isin(["1","Yes","1.0"])) | (df["Functioning Day"] == 1)]
    avg_demand = int(func_df["BikeCount"].mean())
    peak_hour  = int(func_df.groupby("Hour")["BikeCount"].mean().idxmax())
    peak_month_idx = int(func_df.groupby("Month")["BikeCount"].mean().idxmax())
    peak_month = MONTH_NAMES[peak_month_idx - 1]
    best_model = max(MODEL_METRICS, key=lambda m: MODEL_METRICS[m]["R²"])
    best_r2    = MODEL_METRICS[best_model]["R²"]

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Avg Hourly Demand</div>
        <div class="kpi-value">{avg_demand:,}</div>
        <div class="kpi-sub">bikes / hour (operating days)</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">Peak Hour</div>
        <div class="kpi-value">{peak_hour:02d}:00</div>
        <div class="kpi-sub">highest avg rentals</div>
      </div>
      <div class="kpi-card orange">
        <div class="kpi-label">Peak Month</div>
        <div class="kpi-value">{peak_month}</div>
        <div class="kpi-sub">highest seasonal demand</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Best Model R²</div>
        <div class="kpi-value">{best_r2:.3f}</div>
        <div class="kpi-sub">{best_model}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-label">Best R²</div>
        <div class="kpi-value">0.877</div><div class="kpi-sub">XGBoost</div></div>
      <div class="kpi-card red"><div class="kpi-label">Models Trained</div>
        <div class="kpi-value">{len(MODEL_METRICS)}</div><div class="kpi-sub">regression models</div></div>
      <div class="kpi-card orange"><div class="kpi-label">Features</div>
        <div class="kpi-value">12</div><div class="kpi-sub">weather + temporal</div></div>
      <div class="kpi-card"><div class="kpi-label">Dataset</div>
        <div class="kpi-value">8,760</div><div class="kpi-sub">hourly observations</div></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_explorer, tab_analytics, tab_comparison, tab_about = st.tabs([
    "🤖  Model Explorer",
    "📊  Analytics",
    "📈  Model Comparison",
    "ℹ️  About",
])

# ══════════════════════════════════════════════
#  TAB 1 — ANALYTICS
# ══════════════════════════════════════════════
with tab_analytics:
    if not has_data:
        st.info("📂 Place **SeoulBikeData.csv** in the `data/` folder to unlock the Analytics tab.")
    else:
        func_df = df[(df["Functioning Day"].astype(str).isin(["1","Yes","1.0"])) | (df["Functioning Day"] == 1)]

        # ── Row 1: Hourly + Monthly ───────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⏱ Average Hourly Demand</div>', unsafe_allow_html=True)

            hourly = func_df.groupby(["Hour", "IsWeekend"])["BikeCount"].mean().reset_index()
            hourly["Day Type"] = hourly["IsWeekend"].map({True: "Weekend", False: "Weekday"})

            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig_bg(fig)
            for dtype, color in [("Weekday", BLUE), ("Weekend", RED)]:
                sub = hourly[hourly["Day Type"] == dtype]
                ax.fill_between(sub["Hour"], sub["BikeCount"], alpha=0.15, color=color)
                ax.plot(sub["Hour"], sub["BikeCount"], marker="o", markersize=3,
                        color=color, linewidth=2, label=dtype)
            apply_style(ax, "", "Hour of Day", "Avg Bike Rentals")
            ax.set_xticks(range(0, 24, 2))
            ax.legend(fontsize=8, frameon=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption("Weekdays show a clear AM/PM commute double-peak. Weekends peak mid-afternoon.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📅 Average Monthly Demand</div>', unsafe_allow_html=True)

            monthly = func_df.groupby("Month")["BikeCount"].mean().reset_index()
            monthly["MonthName"] = monthly["Month"].apply(lambda m: MONTH_NAMES[m-1])

            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig_bg(fig)
            colors = [RED if v == monthly["BikeCount"].max() else BLUE for v in monthly["BikeCount"]]
            ax.bar(monthly["MonthName"], monthly["BikeCount"], color=colors, width=0.7, zorder=3)
            ax.yaxis.grid(True, color="#e5e7eb", zorder=0)
            ax.set_axisbelow(True)
            apply_style(ax, "", "", "Avg Bike Rentals")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption("Peak demand in June–September. Winter months (Dec–Feb) see the sharpest drop.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 2: Heatmap ────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🗓 Demand Heatmap · Hour × Month</div>', unsafe_allow_html=True)

        pivot = func_df.groupby(["Month","Hour"])["BikeCount"].mean().unstack(level=1)
        pivot.index = [MONTH_NAMES[i-1] for i in pivot.index]

        fig, ax = plt.subplots(figsize=(13, 4))
        fig_bg(fig)
        sns.heatmap(
            pivot, ax=ax,
            cmap=sns.light_palette(BLUE, as_cmap=True),
            linewidths=0.3, linecolor="#f0f0f0",
            cbar_kws={"label": "Avg Rentals", "shrink": 0.8},
        )
        ax.set_xlabel("Hour of Day", fontsize=9, color=GREY)
        ax.set_ylabel("")
        ax.tick_params(colors=GREY, labelsize=8)
        ax.set_title("")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Summer mornings and evenings show the highest demand. Cold winter nights approach zero.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 3: Weather Sensitivity ────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🌤 Weather Sensitivity</div>', unsafe_allow_html=True)

        wcol1, wcol2, wcol3 = st.columns(3)

        with wcol1:
            bins = [-20, 0, 10, 20, 30, 40]
            labels = ["< 0°C", "0–10°C", "10–20°C", "20–30°C", "> 30°C"]
            func_df = func_df.copy()
            func_df["TempBin"] = pd.cut(func_df["Temperature(°C)"], bins=bins, labels=labels)
            temp_g = func_df.groupby("TempBin", observed=True)["BikeCount"].mean()
            fig, ax = plt.subplots(figsize=(4, 3))
            fig_bg(fig)
            colors_t = [BLUE if v < temp_g.max() else RED for v in temp_g.values]
            ax.bar(temp_g.index, temp_g.values, color=colors_t, width=0.6, zorder=3)
            ax.yaxis.grid(True, color="#e5e7eb", zorder=0); ax.set_axisbelow(True)
            apply_style(ax, "Temperature", "", "Avg Rentals")
            plt.xticks(rotation=20, ha="right", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with wcol2:
            rain_g = func_df.groupby(func_df["Rainfall(mm)"] > 0)["BikeCount"].mean()
            rain_g.index = ["No Rain", "Raining"]
            fig, ax = plt.subplots(figsize=(4, 3))
            fig_bg(fig)
            ax.bar(rain_g.index, rain_g.values, color=[BLUE, RED], width=0.5, zorder=3)
            ax.yaxis.grid(True, color="#e5e7eb", zorder=0); ax.set_axisbelow(True)
            apply_style(ax, "Rain Impact", "", "Avg Rentals")
            pct = (1 - rain_g["Raining"] / rain_g["No Rain"]) * 100
            ax.annotate(f"−{pct:.0f}% on rainy hours", xy=(1, rain_g["Raining"]),
                        xytext=(0.5, rain_g["Raining"] + 50),
                        fontsize=7, color=RED, ha="center")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with wcol3:
            hol_map = {0: "Working Day", 1: "Holiday"}
            hol_g = func_df.groupby("Holiday")["BikeCount"].mean()
            hol_g.index = [hol_map.get(i, i) for i in hol_g.index]
            fig, ax = plt.subplots(figsize=(4, 3))
            fig_bg(fig)
            ax.bar(hol_g.index, hol_g.values, color=[BLUE, ACCENT], width=0.5, zorder=3)
            ax.yaxis.grid(True, color="#e5e7eb", zorder=0); ax.set_axisbelow(True)
            apply_style(ax, "Holiday Effect", "", "Avg Rentals")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        st.caption("Demand rises steeply with temperature. Demands significantly drops with rain. Commuter patterns mean working days outpace holidays.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 4: Season box plots ────────────────
        # Seoul dataset: Seasons col is int (1=Winter,2=Spring,3=Summer,4=Autumn)
        season_int_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
        season_str_valid = {"Winter", "Spring", "Summer", "Autumn"}
        if "Season" in func_df.columns:
            raw_seasons = func_df["Season"]
            if pd.api.types.is_numeric_dtype(raw_seasons):
                func_df = func_df.copy()
                func_df["SeasonName"] = raw_seasons.map(season_int_map)
            else:
                # already strings — normalise capitalisation
                func_df = func_df.copy()
                func_df["SeasonName"] = raw_seasons.str.strip().str.capitalize()
            # Drop rows where mapping failed
            func_df = func_df[func_df["SeasonName"].isin(season_str_valid)]

        if "Season" in func_df.columns and "SeasonName" in func_df.columns and not func_df.empty:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🍂 Demand Distribution by Season</div>', unsafe_allow_html=True)
            season_order = ["Spring","Summer","Autumn","Winter"]

            fig, ax = plt.subplots(figsize=(10, 3.5))
            fig_bg(fig)
            season_colors = [ACCENT, RED, BLUE, GREY]
            season_data   = [func_df.loc[func_df["SeasonName"] == s, "BikeCount"].dropna().values
                             for s in season_order]
            bp = ax.boxplot(season_data, patch_artist=True, widths=0.5,
                            medianprops=dict(color=WHITE, linewidth=2),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2),
                            flierprops=dict(marker=".", markersize=3, alpha=0.3))
            for patch, color in zip(bp["boxes"], season_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(season_order) + 1))
            ax.set_xticklabels(season_order)
            apply_style(ax, "", "", "Bike Rentals / Hour")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption("Summer shows the highest median demand and widest spread while winter shows the narrowest.")
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 — MODEL EXPLORER
# ══════════════════════════════════════════════
with tab_explorer:
    if not models:
        st.error("No model files found in `models/`. Ensure your `.pkl` files are present.")
        st.stop()

    st.markdown("#### Configure inputs and get an instant demand forecast")

    # ── Compute data-derived averages for preset button ─────────────────
    if has_data:
        op = df[(df["Functioning Day"].astype(str).isin(["1","Yes","1.0"])) | (df["Functioning Day"] == 1)]
        # Use conditions that match a typical high-demand hour (18:00, June, no rain)
        avg_row = op[
            (op["Hour"] == 18) &
            (op["Month"] == 6) &
            (op["Rainfall(mm)"] == 0) &
            (op["Snowfall (cm)"] == 0)
        ].mean(numeric_only=True)

        DATA_DEFAULTS = {
            "temp":  round(float(avg_row.get("Temperature(°C)",      22.0)), 1),
            "dew":   round(float(avg_row.get("Dew point temperature(°C)", 15.0)), 1),
            "humid": round(float(avg_row.get("Humidity(%)",           60.0)), 1),
            "wind":  round(float(avg_row.get("Wind speed (m/s)",       1.5)), 1),
            "vis":   round(float(avg_row.get("Visibility (10m)",      1600.0)), 0),
            "solar": round(float(avg_row.get("Solar Radiation (MJ/m2)", 0.5)), 2),
        }
    else:
        DATA_DEFAULTS = {
            "temp": 22.0, "dew": 15.0, "humid": 60.0,
            "wind": 1.5,  "vis": 1600.0, "solar": 0.5,
        }

    # Use session state so preset button can update sliders
    SS = st.session_state
    if "d_temp"  not in SS: SS.d_temp  = DATA_DEFAULTS["temp"]
    if "d_dew"   not in SS: SS.d_dew   = DATA_DEFAULTS["dew"]
    if "d_humid" not in SS: SS.d_humid = DATA_DEFAULTS["humid"]
    if "d_wind"  not in SS: SS.d_wind  = DATA_DEFAULTS["wind"]
    if "d_vis"   not in SS: SS.d_vis   = float(DATA_DEFAULTS["vis"])
    if "d_solar" not in SS: SS.d_solar = DATA_DEFAULTS["solar"]
    if "d_rain"  not in SS: SS.d_rain  = 0.0
    if "d_snow"  not in SS: SS.d_snow  = 0.0
    if "d_hour"  not in SS: SS.d_hour  = 18
    if "d_month" not in SS: SS.d_month = 6


    # ── Input form ────────────────────────────
    with st.form("predictor_form"):
        st.markdown("**🌦 Weather Conditions**")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        temp   = r1c1.slider("Temperature (°C)",        -17.8, 39.4,  SS.d_temp)
        dew    = r1c2.slider("Dew Point Temp (°C)",     -30.6, 27.2,  SS.d_dew)
        humid  = r1c3.slider("Humidity (%)",              0.0, 98.0,  SS.d_humid)
        wind   = r1c4.slider("Wind Speed (m/s)",          0.0,  7.4,  SS.d_wind)

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        solar  = r2c1.slider("Solar Radiation (MJ/m²)",  0.0,  3.52, SS.d_solar)
        vis    = r2c2.slider("Visibility (10m)",         27.0, 2000.0, SS.d_vis)
        rain   = r2c3.slider("Rainfall (mm)",             0.0, 35.0,  SS.d_rain)
        snow   = r2c4.slider("Snowfall (cm)",             0.0,  8.8,  SS.d_snow)

        st.markdown("**📅 Date & Time**")
        r3c1, r3c2, r3c3, r3c4, _ = st.columns(5)
        hour    = r3c1.selectbox("Hour",            range(24),   index=SS.d_hour)
        month   = r3c2.selectbox("Month",           range(1, 13), index=SS.d_month - 1)
        holiday = r3c3.selectbox("Holiday",         [0, 1], format_func=lambda x: "Yes" if x else "No")
        func_d  = r3c4.selectbox("Functioning Day", [1, 0], format_func=lambda x: "Yes" if x else "No")

        st.markdown("**🤖 Model**")
        selected = st.selectbox("Select Model", list(models.keys()),
                                 index=list(models.keys()).index("Random Forest")
                                 if "Random Forest" in models else 0)

        submitted = st.form_submit_button("🔍  Predict Demand", use_container_width=True)

    user_inputs = {
        "Hour": hour, "Temperature(°C)": temp, "Humidity(%)": humid,
        "Wind speed (m/s)": wind, "Visibility (10m)": vis,
        "Dew point temperature(°C)": dew, "Solar Radiation (MJ/m2)": solar,
        "Rainfall(mm)": rain, "Snowfall (cm)": snow,
        "Holiday": holiday, "Functioning Day": func_d, "Month": month,
    }

    if submitted:
        model_obj = models[selected]
        pred_val, log_applied = predict(model_obj, user_inputs)


        # Badge
        if pred_val < 500:
            badge_cls, icon, label = "pred-low",  "🟢", "Low Demand"
        elif pred_val < 1500:
            badge_cls, icon, label = "pred-mid",  "🟡", "Moderate Demand"
        else:
            badge_cls, icon, label = "pred-high", "🔴", "High Demand"

        # Mean demand baseline for context (operating days only)
        mean_demand = None
        if has_data:
            op = df[(df["Functioning Day"].astype(str).isin(["1","Yes","1.0"])) | (df["Functioning Day"] == 1)]
            if "Hour" in op.columns and "BikeCount" in op.columns:
                mean_demand = op[op["Hour"] == hour]["BikeCount"].mean()

        pc1, pc2, pc3 = st.columns([1.5, 1, 1])
        with pc1:
            st.markdown(f"""
            <div class="section-card" style="text-align:center; padding:2rem;">
              <div class="kpi-label">Predicted Bike Count · {selected}</div>
              <div class="pred-badge {badge_cls}">{icon}  {int(pred_val):,} bikes</div>
              <div style="font-size:0.9rem; color:{GREY};">{label}</div>
              {f'<div style="font-size:0.78rem;color:#6B7280;margin-top:0.4rem;">Historical avg at {hour:02d}:00 on operating days: <b>{int(mean_demand):,}</b> bikes</div>' if mean_demand is not None else ''}
            </div>
            """, unsafe_allow_html=True)

        with pc2:
            r2 = MODEL_METRICS.get(selected, {}).get("R²", "—")
            mse = MODEL_METRICS.get(selected, {}).get("MSE", "—")
            st.metric("Model R²",  f"{r2:.4f}" if isinstance(r2, float) else r2)
            st.metric("Model MSE", f"{mse:,}"  if isinstance(mse, int)  else mse)

        with pc3:
            if selected != "Random Forest" and "Random Forest" in models:
                rf_pred, _ = predict(models["Random Forest"], user_inputs)
                diff    = pred_val - rf_pred
                sign    = "+" if diff >= 0 else ""
                st.metric("Random Forest", f"{int(rf_pred):,}",
                           delta=f"{sign}{int(diff)} vs RF")

        # ── Hourly Forecast ──────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📉 Hourly Forecast for Configured Day</div>', unsafe_allow_html=True)

        hourly_inputs = []
        for h in range(24):
            row = user_inputs.copy()
            row["Hour"] = h
            hourly_inputs.append(row)

        hourly_df   = pd.DataFrame(hourly_inputs).reindex(columns=FEATURE_ORDER)
        hourly_raw  = model_obj.predict(hourly_df).astype(float)
        # Apply same inverse-log logic used for single prediction
        if log_applied:
            hourly_pred = np.maximum(np.expm1(hourly_raw), 0)
        else:
            hourly_pred = np.maximum(hourly_raw, 0)

        fig, ax = plt.subplots(figsize=(11, 4))
        fig_bg(fig)
        ax.fill_between(range(24), hourly_pred, alpha=0.12, color=BLUE)
        ax.plot(range(24), hourly_pred, marker="o", markersize=5,
                color=BLUE, linewidth=2.5, zorder=5)

        # Highlight current hour
        ax.axvline(hour, color=RED, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.annotate(f"Hour {hour:02d}\n{int(hourly_pred[hour]):,}",
                    xy=(hour, hourly_pred[hour]),
                    xytext=(hour + 0.8, hourly_pred[hour] + 30),
                    fontsize=8, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

        apply_style(ax, f"Predicted Demand Across 24 Hours  ({selected})",
                    "Hour of Day", "Predicted Bike Rentals")
        ax.set_xticks(range(0, 24))
        ax.yaxis.grid(True, color="#e5e7eb")
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── All-model comparison for this input ──
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚖️ All-Model Predictions for This Input</div>', unsafe_allow_html=True)

        preds_all = {m: predict(models[m], user_inputs)[0] for m in models}
        preds_df  = pd.DataFrame({"Model": list(preds_all.keys()),
                                   "Predicted Bikes": list(preds_all.values())})
        preds_df  = preds_df.sort_values("Predicted Bikes", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig_bg(fig)
        bar_colors = [RED if m == selected else BLUE for m in preds_df["Model"]]
        ax.barh(preds_df["Model"], preds_df["Predicted Bikes"],
                color=bar_colors, height=0.55, zorder=3)
        ax.xaxis.grid(True, color="#e5e7eb", zorder=0)
        ax.set_axisbelow(True)
        for i, (_, row) in enumerate(preds_df.iterrows()):
            ax.text(row["Predicted Bikes"] + 5, i,
                    f"{int(row['Predicted Bikes']):,}", va="center", fontsize=8, color=GREY)
        apply_style(ax, "", "Predicted Bike Count", "")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Feature Importance (tree models) ─────
        best_estimator = None
        if selected == "Random Forest":
            rf_gs = models["Random Forest"]
            best_estimator = rf_gs.best_estimator_ if hasattr(rf_gs, "best_estimator_") else rf_gs
        elif selected in ("XGBoost", "LightGBM"):
            best_estimator = model_obj

        if best_estimator is not None and hasattr(best_estimator, "feature_importances_"):
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">🎯 Feature Importance · {selected}</div>', unsafe_allow_html=True)

            fi = best_estimator.feature_importances_
            fi_df = (pd.DataFrame({"Feature": FEATURE_ORDER, "Importance": fi})
                       .sort_values("Importance", ascending=True))

            fig, ax = plt.subplots(figsize=(8, 4))
            fig_bg(fig)
            bar_c = [RED if v == fi_df["Importance"].max() else BLUE for v in fi_df["Importance"]]
            ax.barh(fi_df["Feature"], fi_df["Importance"], color=bar_c, height=0.6, zorder=3)
            ax.xaxis.grid(True, color="#e5e7eb", zorder=0)
            ax.set_axisbelow(True)
            apply_style(ax, "", "Importance Score", "")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Sensitivity plots ─────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📐 Feature Sensitivity  (all else held constant)</div>', unsafe_allow_html=True)

        sens_features = ["Temperature(°C)", "Humidity(%)", "Wind speed (m/s)",
                         "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Visibility (10m)"]
        n_cols   = 3
        n_rows   = int(np.ceil(len(sens_features) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows))
        fig_bg(fig)
        axes = axes.flatten()

        for i, feat in enumerate(sens_features):
            fmin, fmax, _ = FEATURES_INFO[feat]
            frange = np.linspace(fmin, fmax, 60)
            tmp = pd.DataFrame([user_inputs] * 60).reindex(columns=FEATURE_ORDER)
            tmp[feat] = frange
            sens_raw = model_obj.predict(tmp).astype(float)
            preds_s = np.maximum(np.expm1(sens_raw) if log_applied else sens_raw, 0)

            axes[i].fill_between(frange, preds_s, alpha=0.12, color=BLUE)
            axes[i].plot(frange, preds_s, color=BLUE, linewidth=2)
            axes[i].axvline(user_inputs[feat], color=RED, linewidth=1.2,
                             linestyle="--", alpha=0.8)
            apply_style(axes[i], feat, "", "Predicted Bikes")
            axes[i].yaxis.grid(True, color="#e5e7eb")
            axes[i].set_axisbelow(True)

        for j in range(len(sens_features), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Red dashed line = your current input value.")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab_comparison:
    st.markdown("#### Head-to-head performance on the held-out test set")

    metrics_df = (
        pd.DataFrame(MODEL_METRICS)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .sort_values("R²", ascending=False)
        .reset_index(drop=True)
    )
    metrics_df.index += 1  # rank from 1

    # ── Leaderboard ───────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Model Leaderboard</div>', unsafe_allow_html=True)

    def highlight_best(s):
        if s.name == "R²":
            return ["background-color:#dbeafe; font-weight:700"
                    if v == s.max() else "" for v in s]
        if s.name in ("MSE", "MAE"):
            return ["background-color:#dbeafe; font-weight:700"
                    if v == s.min() else "" for v in s]
        return [""] * len(s)

    styled = (
        metrics_df.style
        .apply(highlight_best, axis=0)
        .format({"R²": "{:.4f}", "MSE": "{:,.0f}", "MAE": "{:,.0f}"})
        .set_properties(**{"font-family": "'IBM Plex Mono', monospace", "font-size": "0.85rem"})
    )
    st.dataframe(styled, use_container_width=True, height=280)
    st.caption("Highlighted cells = best value per metric. Metrics computed on identical held-out test set.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── R² + MSE side-by-side ─────────────────
    cmp1, cmp2 = st.columns(2)

    with cmp1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">R² Score  (higher = better)</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5.5, 4))
        fig_bg(fig)
        bar_colors = [RED if v == metrics_df["R²"].max() else BLUE
                      for v in metrics_df["R²"]]
        ax.barh(metrics_df["Model"][::-1], metrics_df["R²"][::-1],
                color=bar_colors[::-1], height=0.6, zorder=3)
        ax.axvline(0.8, color=GREY, linewidth=1, linestyle="--", alpha=0.5)
        ax.xaxis.grid(True, color="#e5e7eb", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1)
        for i, (_, row) in enumerate(metrics_df[::-1].reset_index().iterrows()):
            ax.text(row["R²"] + 0.005, i, f'{row["R²"]:.4f}', va="center",
                    fontsize=8, color=GREY)
        apply_style(ax, "", "R² Score", "")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cmp2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">MSE  (lower = better)</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5.5, 4))
        fig_bg(fig)
        bar_colors2 = [RED if v == metrics_df["MSE"].min() else BLUE
                       for v in metrics_df["MSE"]]
        ax.barh(metrics_df["Model"][::-1], metrics_df["MSE"][::-1],
                color=bar_colors2[::-1], height=0.6, zorder=3)
        ax.xaxis.grid(True, color="#e5e7eb", zorder=0)
        ax.set_axisbelow(True)
        for i, (_, row) in enumerate(metrics_df[::-1].reset_index().iterrows()):
            ax.text(row["MSE"] + 500, i, f'{row["MSE"]:,.0f}', va="center",
                    fontsize=7.5, color=GREY)
        apply_style(ax, "", "Mean Squared Error", "")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Key takeaways ─────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Key Takeaways</div>', unsafe_allow_html=True)
    st.markdown(f"""
    - **Gradient boosting models** (XGBoost, LightGBM) achieve the best overall performance,
      with R² above **0.87**, meaning they explain roughly 87% of variance in hourly demand.
    - **Random Forest** is a close competitor and provides richer interpretability via
      feature importances.
    - **Linear models** (Linear Regression, Ridge, Lasso) plateau at R² ≈ **0.41**, confirming
      that bike demand has strong non-linear relationships with weather and time.
    - **KNN** sits in the middle. It is better than linear, but unable to generalize as well as
      ensemble and boosting approaches.
    - The single most impactful modeling improvement came from switching from linear to
      tree-based methods, not from extensive hyperparameter tuning.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════
with tab_about:
    a1, a2 = st.columns([2, 1])

    with a1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📌 Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        Seoul operates one of the world's largest public bicycle-sharing schemes. Accurately
        forecasting hourly demand lets city planners **pre-position bikes**, optimize maintenance
        schedules, and improve rider experience.

        This project builds and compares **seven regression models** from simple linear
        baselines to gradient-boosted trees. They are trained on two full years of hourly sensor data
        covering weather, time-of-day, and calendar features.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Key Findings</div>', unsafe_allow_html=True)
        st.markdown(f"""
        | Finding | Detail |
        |---|---|
        | **Top predictor** | Temperature (demand rises sharply above 10 °C) |
        | **Commuter double-peak** | Strong 08:00 and 18:00 spikes on weekdays |
        | **Seasonal swing** | Summer demand ~3x higher than winter |
        | **Rain effect** | Rentals drop sharply during rainfall hours |
        | **Best model** | XGBoost (R² {MODEL_METRICS['XGBoost']['R²']:.3f}, MSE {MODEL_METRICS['XGBoost']['MSE']:,}) |
        | **Linear ceiling** | Linear models cap at R² ≈ 0.41, confirming non-linearity |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🛠 Tech Stack</div>', unsafe_allow_html=True)

        tags = ["Python 3.11", "Streamlit", "Scikit-learn", "XGBoost",
                "LightGBM", "Pandas", "NumPy", "Matplotlib", "Seaborn",
                "GridSearchCV"]
        st.markdown(" ".join(f'<span class="tag">{t}</span>' for t in tags),
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📦 Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        **Seoul Bike Sharing Demand**
        UCI Machine Learning Repository

        - **8,760** hourly observations
        - **2 years** of data (2017-2018)
        - **14 raw features** (weather + calendar)
        - Target: `Rented Bike Count`

        [🔗 View on UCI](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🤖 Models Trained</div>', unsafe_allow_html=True)
        for m, metrics in sorted(MODEL_METRICS.items(), key=lambda x: -x[1]["R²"]):
            medal = "🥇" if metrics["R²"] == max(v["R²"] for v in MODEL_METRICS.values()) else "  "
            st.markdown(f"{medal} **{m}** — R² `{metrics['R²']:.4f}`")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Kenny Luu</div>', unsafe_allow_html=True)
        st.markdown("""
        Built as a portfolio project demonstrating end-to-end ML:
        data exploration → feature engineering → model training →
        hyperparameter tuning → interactive deployment.

        ---
        🔗 [GitHub](https://github.com/kennyluu130/Seoul-Bike-Demand)
        · [LinkedIn](http://linkedin.com/in/kennyluu130)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Citation
    st.divider()
    