"""AutoValue AI — Smart Car Price Predictor (Streamlit). Run: streamlit run app.py"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import (
    ALLOWED_FUELS,
    ALLOWED_OWNERS,
    ALLOWED_SELLER_TYPES,
    ALLOWED_TRANSMISSIONS,
    DATA_PATH,
    MAX_YEAR,
    METRICS_PATH,
    MIN_YEAR,
)
from src.data_loader import load_clean
from src.predict import feature_importance, load_artifact, predict_price, validate_inputs

st.set_page_config(
    page_title="AutoValue AI — Smart Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY, ACCENT = "#0ea5e9", "#6366f1"


def is_dark() -> bool:
    """Detect Streamlit's active base theme (updates on rerun when the user
    toggles light/dark in the Settings menu)."""
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


DARK_MODE = is_dark()
INK = "#f1f5f9" if DARK_MODE else "#0f172a"
MUTED = "#94a3b8" if DARK_MODE else "#64748b"
CARD_BG = "rgba(255,255,255,.06)" if DARK_MODE else "#ffffff"
CARD_BORDER = "rgba(255,255,255,.14)" if DARK_MODE else "#e2e8f0"
STEP_BG = "rgba(255,255,255,.04)" if DARK_MODE else "#f8fafc"
EDGE = "#e2e8f0" if DARK_MODE else "#0f172a"
HL = "#14532d" if DARK_MODE else "#dcfce7"

sns.set_theme(style="darkgrid" if DARK_MODE else "whitegrid")

st.markdown(f"""
<style>
.hero {{background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 45%,#0ea5e9 100%);
  color:#fff;border-radius:18px;padding:2.2rem 2rem;margin-bottom:1.2rem;}}
.hero h1 {{margin:0;font-size:2.1rem;}}
.hero p {{opacity:.88;margin:.5rem 0 0;}}
.kpi {{background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:14px;
  padding:1rem 1.1rem;box-shadow:0 2px 10px rgba(15,23,42,.06);}}
.kpi .v {{font-size:1.45rem;font-weight:800;color:{INK};}}
.kpi .l {{font-size:.8rem;color:{MUTED};text-transform:uppercase;letter-spacing:.06em;}}
.step {{background:{STEP_BG};border:1px solid {CARD_BORDER};border-radius:14px;
  padding:1rem 1.1rem;height:100%;color:{INK};}}
.result {{background:linear-gradient(120deg,#059669,#0ea5e9);color:#fff;border-radius:18px;
  padding:1.8rem;text-align:center;margin:1rem 0;}}
.result .p {{font-size:2.6rem;font-weight:900;margin:.2rem 0;}}
.result .s {{opacity:.9;}}
.stButton>button {{border-radius:10px;font-weight:700;}}
footer {{visibility:hidden;}}
.foot {{text-align:center;color:{MUTED};font-size:.8rem;margin-top:2rem;}}
</style>
""", unsafe_allow_html=True)


def fig_ax(*args, **kwargs):
    """Create a matplotlib figure whose background and text follow the
    active Streamlit theme."""
    fig, ax = fig_ax(*args, **kwargs)
    bg = "#0e1117" if DARK_MODE else "#ffffff"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    return fig, ax


def inr(x) -> str:
    return f"₹ {x:,.0f}"


@st.cache_resource
def get_artifact():
    return load_artifact()


@st.cache_data
def get_data():
    try:
        return load_clean(DATA_PATH)
    except Exception:
        return None


@st.cache_data
def get_metrics():
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None


def sidebar():
    st.sidebar.markdown("## 🚗 AutoValue AI")
    st.sidebar.caption("Smart used-car price estimates (INR)")
    nav = st.sidebar.radio(
        "Navigate", ["🏠 Home", "💰 Predict Price", "📊 Model Insights"], key="nav"
    )
    st.sidebar.divider()
    try:
        art = get_artifact()
        st.sidebar.success(f"Model ready: {art.get('model_name', 'pipeline')}")
    except Exception as exc:
        st.sidebar.error("Model not trained")
        st.sidebar.caption(str(exc)[:120])
    df = get_data()
    if df is not None:
        st.sidebar.caption(
            f"{len(df):,} listings · {df['brand'].nunique()} brands · "
            f"median {inr(df['selling_price'].median())}"
        )
    st.sidebar.divider()
    st.sidebar.caption("Estimates are indicative, not guaranteed resale values.")
    return nav


def kpi(col, label, value):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div>"
                 f"<div class='v'>{value}</div></div>", unsafe_allow_html=True)


def page_home():
    df, m = get_data(), get_metrics()
    st.markdown("<div class='hero'><h1>🚗 AutoValue AI — Smart Car Price Predictor</h1>"
                "<p>Data-driven used-car price estimates trained on historical "
                "CarDekho listings. Enter specs, get an instant ₹ estimate with "
                "model-backed context.</p></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Listings analysed", f"{len(df):,}" if df is not None else "—")
    kpi(c2, "Median price", inr(df["selling_price"].median()) if df is not None else "—")
    kpi(c3, "Best model", m["model"] if m else "—")
    best_r2 = m["results"][m["model"]]["r2"] if m else None
    kpi(c4, "Test R²", f"{best_r2:.3f}" if best_r2 is not None else "—")

    st.subheader("How it works")
    s1, s2, s3 = st.columns(3)
    s1.markdown("<div class='step'><b>1 · Describe the car</b><br>Brand, year, kilometres, "
                "fuel, transmission, seller & ownership.</div>", unsafe_allow_html=True)
    s2.markdown("<div class='step'><b>2 · ML pipeline</b><br>The same scaling + encoding "
                "used in training transforms your input for the Random Forest model.</div>",
                unsafe_allow_html=True)
    s3.markdown("<div class='step'><b>3 · Instant estimate</b><br>A prominent ₹ price, "
                "typical range, and the factors most associated with it.</div>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if m:
            st.subheader("Model leaderboard (held-out test)")
            lb = pd.DataFrame(m["results"]).T.rename(
                columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²"})
            st.dataframe(lb.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                                           "R²": "{:.3f}"}).highlight_max(
                subset=["R²"], color=HL), use_container_width=True)
    with col_b:
        st.subheader("Try it")
        st.write("Get your first estimate in under 30 seconds.")
        if st.button("💰 Open prediction form", type="primary", use_container_width=True):
            st.session_state["nav"] = "💰 Predict Price"
            st.rerun()
    st.warning("Prices reflect listings up to 2020 — actual resale depends on condition, "
               "location and demand, which the model may not capture.")


def page_predict():
    st.title("💰 Predict Selling Price")
    try:
        artifact = get_artifact()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.info("Train the model first: `python -m src.train`")
        return
    df, m = get_data(), get_metrics()
    brands = sorted(df["brand"].unique().tolist()) if df is not None else ["Maruti"]
    mae = m["results"][m["model"]]["mae"] if m else 160_000
    median_all = df["selling_price"].median() if df is not None else None

    with st.form("predict_form"):
        st.markdown("#### 🚘 Vehicle")
        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox("Brand", brands,
                                 help="Derived from the model/variant name")
            model_detail = st.text_input("Model / variant (optional)",
                                         placeholder="e.g. Swift VDI, Creta SX")
            year = st.slider("Manufacturing year", MIN_YEAR, MAX_YEAR, 2015,
                             help="Newer cars are typically associated with higher prices")
        with c2:
            km_driven = st.number_input("Kilometers driven", 0, 2_000_000, 50_000,
                                        step=5_000, help="Higher mileage is typically "
                                        "associated with lower prices")
            fuel = st.selectbox("Fuel type", ALLOWED_FUELS)
            transmission = st.radio("Transmission", ALLOWED_TRANSMISSIONS,
                                    horizontal=True)
        st.markdown("#### 🧾 Sale context")
        c3, c4 = st.columns(2)
        with c3:
            seller_type = st.selectbox("Seller type", ALLOWED_SELLER_TYPES)
        with c4:
            owner = st.selectbox("Ownership history", ALLOWED_OWNERS)
        submitted = st.form_submit_button("✨ Predict Price", type="primary",
                                          use_container_width=True)

    if not submitted:
        st.info("Fill the form above and hit **Predict Price**.")
        return
    errors = validate_inputs(year, km_driven, fuel, seller_type, transmission, owner)
    if errors:
        for e in errors:
            st.error(e)
        return
    car_name = f"{brand} {model_detail}".strip()
    try:
        price = predict_price(car_name, year, km_driven, fuel, seller_type,
                              transmission, owner, artifact)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    lo, hi = max(price - mae, 0), price + mae
    st.markdown(f"<div class='result'><div class='s'>Estimated selling price · "
                f"{car_name} ({year})</div><div class='p'>{inr(price)}</div>"
                f"<div class='s'>Typical range {inr(lo)} – {inr(hi)} "
                f"(± test MAE)</div></div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    if median_all:
        diff = (price - median_all) / median_all * 100
        d1.metric("vs market median", f"{diff:+.1f}%", inr(median_all))
    d2.metric("Model", artifact.get("model_name", "—"))
    d3.metric("Test MAE", inr(mae))

    st.caption("Associated factors (not causal): age and kilometres usually weigh most; "
               "brand, fuel, transmission and ownership adjust the estimate.")
    names, importances = feature_importance(artifact)
    if names is not None:
        top = pd.DataFrame({"feature": names, "importance": importances}
                           ).sort_values("importance", ascending=False).head(12)
        fig, ax = fig_ax(figsize=(8, 4.5))
        sns.barplot(data=top, x="importance", y="feature", ax=ax,
                    color=PRIMARY, edgecolor=EDGE)
        ax.set_title("Top factors associated with the prediction")
        ax.set_xlabel("Importance"); ax.set_ylabel("")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)


def page_insights():
    st.title("📊 Model Insights")
    df, m = get_data(), get_metrics()
    if df is None:
        st.info("Dataset not found.")
        return
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Listings", f"{len(df):,}")
    kpi(c2, "Median price", inr(df["selling_price"].median()))
    kpi(c3, "Year span", f"{df['year'].min()}–{df['year'].max()}")
    kpi(c4, "Selected model", m["model"] if m else "—")

    tab_perf, tab_trends, tab_data = st.tabs(
        ["🏆 Performance", "📈 Price trends", "🗂 Data"])

    with tab_perf:
        if not m:
            st.info("No metrics yet. Run `python -m src.train`.")
        else:
            tbl = pd.DataFrame(m["results"]).T.rename(
                columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²"})
            st.dataframe(tbl.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                                            "R²": "{:.4f}"}).highlight_max(
                subset=["R²"], color=HL), use_container_width=True)
            st.write(f"Trained on {m['n_train']:,} rows, tested on {m['n_test']:,} "
                     f"rows. The {m['model']} beats the median baseline on all errors.")
            fig, ax = fig_ax(figsize=(8, 3.6))
            r2 = tbl["R²"].drop(index="baseline_median", errors="ignore")
            sns.barplot(x=r2.values, y=r2.index, ax=ax, color=ACCENT,
                        edgecolor=EDGE)
            ax.set_title("R² by model (higher is better)")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with tab_trends:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = fig_ax(figsize=(6, 3.6))
            sns.histplot(df["selling_price"].clip(upper=2_000_000), bins=35,
                         kde=True, ax=ax, color=PRIMARY)
            ax.set_title("Price distribution (capped at ₹20L)")
            ax.set_xlabel("Selling price (₹)")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)
        with c2:
            fig, ax = fig_ax(figsize=(6, 3.6))
            med = df.groupby("year")["selling_price"].median().tail(15)
            ax.plot(med.index, med.values, marker="o", color=ACCENT)
            ax.set_title("Median price by manufacturing year")
            ax.set_ylabel("₹"); fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = fig_ax(figsize=(6, 3.6))
            sns.boxplot(data=df, x="fuel", y="selling_price", ax=ax,
                        showfliers=False, color=PRIMARY)
            ax.set_ylim(0, 2_000_000); ax.set_title("Price by fuel type")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)
        with c4:
            fig, ax = fig_ax(figsize=(6, 3.6))
            sns.boxplot(data=df, x="transmission", y="selling_price", ax=ax,
                        showfliers=False, color=ACCENT)
            ax.set_ylim(0, 2_000_000); ax.set_title("Price by transmission")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)
        fig, ax = fig_ax(figsize=(8, 3.4))
        s = df.sample(min(1500, len(df)), random_state=42)
        ax.scatter(s["km_driven"], s["selling_price"], alpha=.25, s=12,
                   color=PRIMARY)
        ax.set_title("Price vs kilometres driven (sample)")
        ax.set_xlabel("km driven"); ax.set_ylabel("₹")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    with tab_data:
        st.dataframe(df.describe(include="all").T, use_container_width=True)
        fig, ax = fig_ax(figsize=(5, 3.6))
        sns.heatmap(df[["year", "km_driven", "car_age", "selling_price"]].corr(),
                    annot=True, cmap="Blues", ax=ax)
        ax.set_title("Numerical correlations")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)
        with st.expander("Preview rows"):
            st.dataframe(df.sample(10, random_state=1), use_container_width=True)


nav = sidebar()
if nav == "🏠 Home":
    page_home()
elif nav == "💰 Predict Price":
    page_predict()
else:
    page_insights()
st.markdown("<div class='foot'>AutoValue AI · estimates are indicative only</div>",
            unsafe_allow_html=True)
