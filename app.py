import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AgriTech | Price & Yield Decision Engine",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    return joblib.load("yield_model.pkl")

@st.cache_data
def load_price_data():
    try:
        df = pd.read_csv("price_data.csv")
        df["dt"] = pd.to_datetime(df["dt"])
        summary = pd.read_csv("price_summary_recent.csv")
        return df, summary
    except Exception as e:
        st.error(f"Price data load error: {e}")
        return pd.DataFrame(), pd.DataFrame()

yield_model = load_models()
price_df, price_summary = load_price_data()

STATE_LIST = [
    "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra",
    "Madhya Pradesh", "Rajasthan", "Gujarat", "Karnataka",
    "Tamil Nadu", "Andhra Pradesh", "Bihar", "West Bengal"
]
CROP_LIST = ["Wheat", "Rice", "Potato", "Tomato", "Onion"]

STATE_COORDS = {
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Madhya Pradesh": {"lat": 23.2599, "lon": 77.4126},
    "Rajasthan": {"lat": 26.9124, "lon": 75.7873},
    "Gujarat": {"lat": 23.2156, "lon": 72.6369},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
    "Bihar": {"lat": 25.0961, "lon": 85.3131},
    "West Bengal": {"lat": 22.9868, "lon": 87.8550}
}

TRANSPORT_RATE = 1.8  # Rs per quintal per km

def get_recent_prices(state, crop, days=60):
    if price_df.empty:
        return pd.DataFrame(), {}
    mask = (price_df["State"] == state) & (price_df["Commodity"] == crop)
    sub = price_df[mask].copy()
    if sub.empty:
        return pd.DataFrame(), {}
    cutoff = sub["dt"].max() - timedelta(days=days)
    recent = sub[sub["dt"] >= cutoff].sort_values("dt")
    daily = recent.groupby("dt")["Modal_Price"].mean().reset_index()
    daily.columns = ["Date", "Modal_Price"]
    market_stats = (
        recent.groupby("Market")
        .agg(
            latest_price=("Modal_Price", "last"),
            avg_price=("Modal_Price", "mean"),
            n_days=("Modal_Price", "count"),
            last_date=("dt", "max")
        )
        .sort_values("n_days", ascending=False)
        .head(6)
    )
    return daily, market_stats.to_dict("index")

def simple_forecast(daily_series, horizon=14):
    if len(daily_series) < 7:
        last = daily_series["Modal_Price"].iloc[-1] if len(daily_series) > 0 else 2000
        return [last] * horizon, last, last
    y = daily_series["Modal_Price"].values[-30:]
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    trend = coef[0]
    last_price = y[-1]
    forecast = [last_price + trend * (i + 1) for i in range(horizon)]
    forecast = [max(last_price * 0.7, min(last_price * 1.4, p)) for p in forecast]
    return forecast, max(forecast), min(forecast)

def get_summary_row(state, crop):
    if price_summary.empty:
        return None
    row = price_summary[(price_summary["State"] == state) & (price_summary["Commodity"] == crop)]
    if row.empty:
        return None
    return row.iloc[0]

# Sidebar
st.sidebar.title("📍 Farm Inputs")
state = st.sidebar.selectbox("State", STATE_LIST, index=0)
crop = st.sidebar.selectbox("Crop", CROP_LIST, index=0)
farm_area = st.sidebar.number_input("Farm Area (Acres)", min_value=0.5, value=5.0, step=0.5)
st.sidebar.markdown("---")
st.sidebar.caption("Data: Agmarknet historical prices (2023-2025)")
st.sidebar.caption("Weather: Open-Meteo | Yield: XGBoost prototype")

st.title("🌾 AgriTech: Price Forecasting & Yield Decision Engine")
st.markdown("**Smart India Hackathon 2026 Prototype** · Focus: **Market Linkages & Price Discovery for Farmers**")

tab1, tab2, tab3 = st.tabs([
    "📊 Yield Prediction",
    "📈 Mandi Price Intelligence",
    "💡 Smart Sell Recommendation"
])

# TAB 1
with tab1:
    st.header("Predict Crop Yield")
    if st.button(f"⛅ Fetch Live Weather for {state}"):
        coords = STATE_COORDS.get(state)
        if coords:
            try:
                url = (f"https://api.open-meteo.com/v1/forecast?"
                       f"latitude={coords['lat']}&longitude={coords['lon']}"
                       f"&current=temperature_2m,relative_humidity_2m,precipitation")
                r = requests.get(url, timeout=8).json()
                cur = r.get("current", {})
                st.success(f"**{state}** → Temp: **{cur.get('temperature_2m','N/A')}°C** | "
                           f"Humidity: **{cur.get('relative_humidity_2m','N/A')}%** | "
                           f"Precip: **{cur.get('precipitation','N/A')} mm**")
            except Exception:
                st.warning("Could not fetch live weather.")

    st.subheader("Soil & Climate Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        nitrogen = st.slider("Nitrogen (N)", 0, 140, 70)
        phosphorus = st.slider("Phosphorus (P)", 0, 145, 45)
    with c2:
        potassium = st.slider("Potassium (K)", 0, 205, 35)
        ph_level = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
    with c3:
        rainfall = st.slider("Expected Rainfall (mm)", 50, 400, 180)

    if st.button("Calculate Yield", type="primary"):
        input_data = np.array([[nitrogen, phosphorus, potassium, ph_level, rainfall]])
        pred_tons = max(0.3, round(float(yield_model.predict(input_data)[0]), 2))
        q_per_acre = round(pred_tons * 10, 1)
        total_q = round(q_per_acre * farm_area, 1)
        st.success(f"**Predicted Yield:** {q_per_acre} quintals / acre  ({pred_tons} tons/acre)")
        st.info(f"**Total expected harvest** for {farm_area} acres → **{total_q} quintals**")
        with st.expander("Model details & limitations"):
            st.markdown("""
            - **Model**: XGBoost Regressor  
            - **Features**: N, P, K, pH, Rainfall  
            - **Training**: Soil-response curves (prototype)  
            - Does not include temperature stress, variety, irrigation or pests.  
            - Use as directional guidance only.
            """)

# TAB 2
with tab2:
    st.header(f"Mandi Price Intelligence — {crop} in {state}")
    daily, market_stats = get_recent_prices(state, crop, days=90)
    summary_row = get_summary_row(state, crop)

    if daily.empty and summary_row is None:
        st.warning(f"No recent mandi data for **{crop}** in **{state}**. Try another combination.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        if summary_row is not None:
            m1.metric("Recent Avg Modal", f"₹{int(summary_row['avg_modal'])}/q")
            m2.metric("Latest Observed", f"₹{int(summary_row['latest_modal'])}/q")
            m3.metric("Range (recent)", f"₹{int(summary_row['min_p'])} – {int(summary_row['max_p'])}")
            m4.metric("Records", f"{int(summary_row['n_records'])}")
        elif not daily.empty:
            m1.metric("Period Average", f"₹{int(daily['Modal_Price'].mean())}/q")
            m2.metric("Latest Day", f"₹{int(daily['Modal_Price'].iloc[-1])}/q")

        if not daily.empty and len(daily) > 3:
            st.subheader("Recent Price Trend (Modal ₹/quintal)")
            st.line_chart(daily.set_index("Date"), height=280)
            _, peak, trough = simple_forecast(daily, 14)
            st.caption(f"Simple 14-day outlook: Peak ~₹{int(peak)} | Trough ~₹{int(trough)}")

        if market_stats:
            st.subheader("Top Markets (by reporting frequency)")
            rows = [{
                "Market": mkt,
                "Latest Modal ₹": int(s["latest_price"]),
                "Avg ₹ (period)": int(s["avg_price"]),
                "Days Reported": int(s["n_days"]),
                "Last Date": str(s["last_date"].date())
            } for mkt, s in market_stats.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("Data source & method"):
            st.markdown("""
            - Source: Agmarknet / data.gov.in historical mandi prices (2023–2025)  
            - Metric: Modal price (₹ per quintal)  
            - Forecast: Transparent linear trend on recent observations
            """)

# TAB 3
with tab3:
    st.header("💡 Smart Selling Decision Advisor")
    st.markdown("Combines expected harvest with **real mandi prices** and transport cost.")

    daily, market_stats = get_recent_prices(state, crop, days=60)
    summary_row = get_summary_row(state, crop)

    default_q = 18.0 if crop == "Wheat" else (12.0 if crop in ["Potato", "Onion"] else 15.0)
    expected_q_per_acre = st.number_input(
        "Expected yield (quintals / acre) — use value from Tab 1 or edit",
        min_value=1.0, value=default_q, step=0.5
    )
    total_quintals = expected_q_per_acre * farm_area
    st.write(f"**Total produce to sell:** **{total_quintals:.1f} quintals**")

    if not market_stats and summary_row is None:
        st.warning("Insufficient price data. Try another state/crop.")
    else:
        if market_stats:
            markets_sorted = sorted(market_stats.items(), key=lambda x: x[1]["latest_price"], reverse=True)
            local_mkt, local_stats = markets_sorted[-1] if len(markets_sorted) > 1 else markets_sorted[0]
            best_mkt, best_stats = markets_sorted[0]
            price_local = int(local_stats["latest_price"])
            price_best = int(best_stats["latest_price"])
            dist_local, dist_best = 12, 45
        else:
            price_local = int(summary_row["median_modal"])
            price_best = int(summary_row["max_p"] * 0.92)
            local_mkt = f"{state} local mandi"
            best_mkt = f"{state} higher-price mandi"
            dist_local, dist_best = 10, 40

        cost_local = TRANSPORT_RATE * dist_local * total_quintals
        cost_best = TRANSPORT_RATE * dist_best * total_quintals
        net_local = (price_local * total_quintals) - cost_local
        net_best = (price_best * total_quintals) - cost_best
        delta = net_best - net_local

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📍 Local / Nearby Option")
            st.write(f"**Market:** {local_mkt}")
            st.write(f"Distance (approx): **{dist_local} km**")
            st.write(f"Modal price: **₹{price_local} / quintal**")
            st.write(f"Transport cost: ₹{cost_local:,.0f}")
            st.metric("Net Revenue", f"₹{net_local:,.0f}")
        with col_b:
            st.subheader("🚛 Higher-Price Option")
            st.write(f"**Market:** {best_mkt}")
            st.write(f"Distance (approx): **{dist_best} km**")
            st.write(f"Modal price: **₹{price_best} / quintal**")
            st.write(f"Transport cost: ₹{cost_best:,.0f}")
            st.metric("Net Revenue", f"₹{net_best:,.0f}", delta=f"₹{delta:,.0f}")

        st.markdown("---")
        if net_best > net_local * 1.03:
            st.success(
                f"**Recommendation: Go to the higher-price market ({best_mkt}).**  \n"
                f"Extra realisation expected: **₹{delta:,.0f}** (≈ ₹{delta/total_quintals:.0f}/quintal)."
            )
        else:
            st.info(
                f"**Recommendation: Sell at the local market ({local_mkt}).**  \n"
                f"Extra price at distant market does not cover transport cost."
            )

        if not daily.empty and len(daily) >= 10:
            recent_trend = daily["Modal_Price"].iloc[-5:].mean() - daily["Modal_Price"].iloc[-15:-5].mean()
            if recent_trend > 30:
                st.warning(f"Prices rising recently (~₹{recent_trend:.0f}). Consider holding 7–10 days if storage is available.")
            elif recent_trend < -40:
                st.warning("Prices softening. Selling sooner may reduce downside risk.")

        with st.expander("Assumptions"):
            st.markdown(f"""
            - Prices are **real modal prices** from Agmarknet for {state} / {crop}.  
            - Transport cost assumed at **₹{TRANSPORT_RATE}/quintal/km**.  
            - Distances are approximate.  
            - Always verify today’s rate on Agmarknet / e-NAM before moving produce.
            """)

st.markdown("---")
st.caption("AgriTech SIH 2026 Prototype · Market linkages & price discovery · Data: Agmarknet historical")
