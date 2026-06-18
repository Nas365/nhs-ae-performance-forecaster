"""
NHS A&E Four-Hour Performance Forecaster
-----------------------------------------
Loads the most recent 13 months of Type 1 A&E data live from NHS England
(cached for 24 hours), builds XGBoost features, and forecasts next-month
performance for each major A&E department in England.

Falls back to the bundled recent_history.csv snapshot if the live fetch fails.
"""
import base64
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data import fetch_live_data
from src.features import FEATURE_COLS, build_forecast_features

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NHS A&E Four-Hour Forecaster",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Background image → base64 (embedded in CSS so it works on any host)
# Falls back to gradient-only if the file is not yet present.
# ---------------------------------------------------------------------------
def _bg_url() -> str:
    img_path = ROOT / "assets" / "app_background.png"
    if not img_path.exists():
        return ""
    data = base64.b64encode(img_path.read_bytes()).decode()
    return f"url('data:image/png;base64,{data}')"


_BG_URL = _bg_url()
_BG_IMAGE_LAYERS = (
    f"linear-gradient(rgba(21,33,46,0.78), rgba(21,33,46,0.88)), {_BG_URL}"
    if _BG_URL
    else "linear-gradient(135deg, rgba(21,33,46,0.95) 0%, rgba(0,48,135,0.90) 100%)"
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    /* ── Background ──────────────────────────────────────────────────────── */
    .stApp {{
        background-image: {_BG_IMAGE_LAYERS};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    [data-testid="stHeader"]           {{ background: transparent !important; }}
    [data-testid="stAppViewContainer"] {{ background: transparent !important; }}
    [data-testid="stToolbar"]          {{ right: 2rem; }}
    footer                             {{ visibility: hidden; }}

    /* ── Block container padding ─────────────────────────────────────────── */
    .block-container {{ padding-top: 2.2rem !important; padding-bottom: 2rem !important; }}

    /* ── Title area (white text over dark background) ────────────────────── */
    h1 {{
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em;
        text-shadow: 0 2px 14px rgba(0,0,0,0.45);
        margin-bottom: 0.25rem !important;
    }}
    .app-header-sub {{
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.80rem !important;
        letter-spacing: 0.03em;
        margin-top: 0 !important;
        margin-bottom: 1.6rem !important;
    }}
    .app-description {{
        color: rgba(255,255,255,0.87) !important;
        font-size: 1.0rem !important;
        margin-bottom: 0.5rem !important;
    }}

    /* ── Content cards (every st.columns call becomes a white panel) ─────── */
    [data-testid="stHorizontalBlock"] {{
        background: rgba(255, 255, 255, 0.97);
        border-radius: 14px;
        padding: 1.4rem 1.8rem 1.2rem !important;
        box-shadow: 0 6px 36px rgba(0,0,0,0.20);
        margin-bottom: 1.2rem !important;
    }}

    /* Dark body text inside white panels */
    [data-testid="stHorizontalBlock"] p,
    [data-testid="stHorizontalBlock"] li,
    [data-testid="stHorizontalBlock"] .stMarkdown p {{
        color: #1a2332 !important;
    }}
    [data-testid="stHorizontalBlock"] h2,
    [data-testid="stHorizontalBlock"] h3 {{
        color: #003087 !important;
    }}
    [data-testid="stHorizontalBlock"] [data-testid="stCaptionContainer"] p,
    [data-testid="stHorizontalBlock"] small {{
        color: #4a5568 !important;
    }}

    /* ── Divider between the two cards ──────────────────────────────────── */
    hr {{
        border-color: rgba(255,255,255,0.15) !important;
        margin: 0.6rem 0 !important;
    }}

    /* ── Metric card inside the info panel ──────────────────────────────── */
    div[data-testid="metric-container"] {{
        background: #f0f4f9;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }}
    div[data-testid="metric-container"] label {{
        font-size: 0.85rem !important;
        color: #4a5568 !important;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        color: #003087 !important;
    }}

    /* ── Footer (white over background) ─────────────────────────────────── */
    .footer-text {{
        font-size: 0.78rem;
        color: rgba(255,255,255,0.60) !important;
    }}
    .footer-text a {{
        color: rgba(255,255,255,0.60) !important;
        text-decoration: underline;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# NHS colour palette (for Plotly)
NHS_DARK_BLUE = "#003087"
NHS_BLUE = "#005EB8"
NHS_YELLOW = "#FFB81C"
NHS_GREY = "#768692"
NHS_LIGHT_GREY = "#e8edee"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_name(raw: str) -> str:
    """ALL-CAPS NHS trust name → readable mixed case."""
    ALWAYS_UPPER = {"NHS", "FT", "NFT", "MRI", "CT", "A&E", "ICU", "ITU", "GP"}
    ALWAYS_LOWER = {"and", "of", "the", "for", "in", "at", "by", "to"}
    words = raw.strip().title().split()
    result = []
    for i, word in enumerate(words):
        if word.upper() in ALWAYS_UPPER:
            result.append(word.upper())
        elif word.lower() in ALWAYS_LOWER and i > 0:
            result.append(word.lower())
        else:
            result.append(word)
    return " ".join(result)


def trend_description(trend_val: float) -> str:
    if trend_val > 0.02:
        return "rising quickly"
    if trend_val > 0.005:
        return "rising"
    if trend_val < -0.02:
        return "falling sharply"
    if trend_val < -0.005:
        return "falling"
    return "broadly stable"


def season_note(month_num: int) -> str:
    if month_num in (12, 1):
        return "December and January are typically the hardest months for major A&Es."
    if month_num == 2:
        return "February usually remains a pressured month for major A&Es."
    if month_num in (3, 4, 5, 6):
        return "Spring and early summer tend to be the calmest period for A&Es."
    if month_num in (7, 8):
        return "Summer months are generally quieter for major A&E departments."
    return "Autumn typically marks the start of rising A&E pressure heading into winter."


# ---------------------------------------------------------------------------
# Data and model loading  (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "ae_model.joblib")


@st.cache_data(ttl=86400, show_spinner=False)
def load_panel() -> tuple:
    """
    Returns (df, source_label, data_month_str).
    Tries the live NHS England feed first (13 months); falls back to the
    bundled snapshot when the fetch fails or is incomplete.
    """
    try:
        df = fetch_live_data(n_months=13, timeout=20)
        source = "live"
    except Exception:
        df = pd.read_csv(ROOT / "recent_history.csv", parse_dates=["month"])
        df = df.sort_values(["org_code", "month"]).reset_index(drop=True)
        source = "snapshot"

    latest = pd.Timestamp(df["month"].max()).strftime("%B %Y")
    return df, source, latest


# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
model = load_model()

with st.spinner("Loading A&E data…"):
    panel, data_source, data_month = load_panel()

# ---------------------------------------------------------------------------
# Header  (white text over dark background — outside any white panel)
# ---------------------------------------------------------------------------
st.title("NHS A&E Four-Hour Performance Forecaster")
st.markdown(
    "<p class='app-header-sub'>"
    "Independent project &mdash; not affiliated with NHS England &nbsp;&middot;&nbsp; "
    "Data: NHS England, Open Government Licence v3.0"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='app-description'>"
    "Select a major (Type&nbsp;1) A&amp;E department to see its recent performance "
    "trend and a one-month-ahead forecast against the four-hour standard."
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hospital selector card
# ---------------------------------------------------------------------------
hospital_map = (
    panel[["org_code", "org_name"]]
    .drop_duplicates()
    .assign(display=lambda d: d["org_name"].map(fmt_name))
    .sort_values("display")
    .set_index("org_code")["display"]
    .to_dict()
)
codes = list(hospital_map.keys())
display_names = list(hospital_map.values())

sel_col, status_col = st.columns([4, 1])
with sel_col:
    selected_idx = st.selectbox(
        "Select hospital",
        range(len(codes)),
        format_func=lambda i: display_names[i],
    )
selected_code = codes[selected_idx]
selected_name = display_names[selected_idx]

with status_col:
    if data_source == "live":
        st.caption(f"✅ Live data · current to {data_month}")
    else:
        st.caption(f"⚠️ Bundled snapshot · {data_month}")

# ---------------------------------------------------------------------------
# Per-hospital forecast
# ---------------------------------------------------------------------------
hosp_df = panel[panel["org_code"] == selected_code].sort_values("month")
feats = build_forecast_features(hosp_df)

if feats is None:
    st.warning("Not enough history for this hospital to generate a forecast (need ≥ 3 months).")
    st.stop()

X_pred = pd.DataFrame([{col: feats[col] for col in FEATURE_COLS}])
forecast_raw = float(model.predict(X_pred)[0])
forecast = float(np.clip(forecast_raw, 0.0, 1.0))

forecast_month: pd.Timestamp = feats["forecast_month"]
latest_month: pd.Timestamp = feats["latest_month"]
recent_avg = float(hosp_df["within_4hrs"].tail(3).mean())
delta = forecast - recent_avg

# ---------------------------------------------------------------------------
# Chart + metrics card
# ---------------------------------------------------------------------------
chart_col, info_col = st.columns([3, 2], gap="large")

# ── Trend chart ─────────────────────────────────────────────────────────────
with chart_col:
    st.subheader("Recent performance")

    y_vals = hosp_df["within_4hrs"].to_numpy()
    y_min = max(0.0, min(y_vals.min(), forecast) - 0.06)
    y_max = min(1.0, max(y_vals.max(), forecast) + 0.06)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hosp_df["month"],
        y=(y_vals * 100).round(1),
        mode="lines+markers",
        name="Actual monthly performance",
        line=dict(color=NHS_BLUE, width=2.5),
        marker=dict(size=5, color=NHS_BLUE),
        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
    ))

    # Dashed bridge to forecast point
    fig.add_trace(go.Scatter(
        x=[latest_month, forecast_month],
        y=[float(y_vals[-1]) * 100, forecast * 100],
        mode="lines",
        line=dict(color=NHS_YELLOW, width=1.5, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[forecast_month],
        y=[round(forecast * 100, 1)],
        mode="markers",
        name=f"Model forecast — {forecast_month.strftime('%B %Y')}",
        marker=dict(size=13, color=NHS_YELLOW, symbol="diamond",
                    line=dict(color=NHS_DARK_BLUE, width=1.5)),
        hovertemplate=(
            f"Forecast {forecast_month.strftime('%b %Y')}: "
            f"{forecast * 100:.1f}%<extra></extra>"
        ),
    ))

    fig.add_hline(
        y=95,
        line_dash="dot",
        line_color=NHS_GREY,
        line_width=1,
        annotation_text="95% standard",
        annotation_position="right",
        annotation_font_color=NHS_GREY,
        annotation_font_size=10,
    )

    fig.update_layout(
        height=360,
        margin=dict(l=65, r=110, t=38, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                    font=dict(size=13, color="#1a2332")),
        yaxis=dict(
            title=dict(
                text="% seen within 4 hours",
                font=dict(size=11, color="#4a5568"),
            ),
            ticksuffix="%",
            tickfont=dict(size=12, color="#1a2332"),
            range=[y_min * 100, y_max * 100],
            gridcolor="#d4dbe0",
            zeroline=False,
            showline=True,
            linecolor="#c8d0d5",
        ),
        xaxis=dict(
            tickfont=dict(size=12, color="#1a2332"),
            gridcolor="#d4dbe0",
            showline=True,
            linecolor="#c8d0d5",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#1a2332"),
    )

    st.plotly_chart(fig, use_container_width=True)

# ── Metrics + narrative ─────────────────────────────────────────────────────
with info_col:
    st.subheader(f"Forecast: {forecast_month.strftime('%B %Y')}")

    st.metric(
        label=selected_name,
        value=f"{forecast * 100:.1f}%",
        delta=f"{delta * 100:+.1f}pp vs 3-month avg",
        delta_color="normal",
    )

    st.markdown("")

    direction = (
        "above" if delta > 0.005 else
        "below" if delta < -0.005 else
        "in line with"
    )
    trend_desc = trend_description(feats["trend_1"])

    st.markdown(
        f"**Outlook**\n\n"
        f"We forecast **{selected_name}** to see **{forecast * 100:.1f}%** of patients "
        f"within four hours in {forecast_month.strftime('%B %Y')} — "
        f"{direction} their recent three-month average of {recent_avg * 100:.1f}%.\n\n"
        f"Performance has been **{trend_desc}** recently. "
        f"{season_note(forecast_month.month)}"
    )

    st.markdown("---")

    lag_12_str = (
        f"{feats['lag_12'] * 100:.1f}%"
        if not np.isnan(feats["lag_12"])
        else "not available"
    )
    st.markdown(
        f"**Key signals**\n\n"
        f"- **Last month** ({latest_month.strftime('%b %Y')}): {feats['lag_1'] * 100:.1f}%\n"
        f"- **3-month average**: {feats['roll_3'] * 100:.1f}%\n"
        f"- **Same month last year**: {lag_12_str}"
    )

    with st.expander("About this model"):
        st.markdown(
            "The forecast uses an XGBoost model trained on all English Type 1 A&E "
            "departments from April 2023 onwards. Evaluated on six sealed test months: "
            "**mean absolute error 2.9 percentage points** — 10% better than a "
            "persistence baseline (last month = next month).\n\n"
            "The dominant signal is last month's performance (feature importance: 76%), "
            "followed by the 3-month rolling average (11%). A Keras LSTM was tested under "
            "the same time-ordered conditions and was less accurate, so XGBoost was chosen "
            "as the production model.\n\n"
            "Estimates carry roughly ±3 percentage points of uncertainty. For illustrative "
            "purposes only — not for operational planning."
        )

# ---------------------------------------------------------------------------
# Footer  (white text over background, outside white panels)
# ---------------------------------------------------------------------------
st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown(
    "<p class='footer-text'>"
    "Contains public sector information from NHS England, licensed under the "
    "<a href='https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/' "
    "target='_blank'>Open Government Licence v3.0</a>. "
    "Type 1 (major, consultant-led, 24/7) A&amp;E departments only. "
    "Forecasts are statistical estimates; not for operational use."
    "</p>",
    unsafe_allow_html=True,
)
