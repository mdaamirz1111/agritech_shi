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
    layout="wide"
)

# =========================
# LANGUAGE DICTIONARY
# =========================
LANG = {
    "en": {
        "app_title": "🌾 AgriTech: Price Forecasting & Yield Decision Engine",
        "app_sub": "SIH 2026 Prototype · Focus: Market Linkages & Price Discovery for Farmers",
        "sidebar_title": "📍 Farm Inputs",
        "state": "State",
        "crop": "Crop",
        "farm_area": "Farm Area (Acres)",
        "data_note": "Data: Agmarknet 2023-2025 | Weather: Open-Meteo",
        "tab_yield": "📊 Yield Prediction",
        "tab_price": "📈 Mandi Price Intelligence",
        "tab_sell": "💡 Smart Sell Recommendation",
        "yield_header": "Predict Crop Yield",
        "fetch_weather": "⛅ Fetch Live Weather for",
        "soil_header": "Soil & Climate Inputs",
        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "ph": "Soil pH",
        "rainfall": "Expected Rainfall (mm)",
        "calc_yield": "Calculate Yield",
        "pred_yield": "Predicted Yield",
        "q_per_acre": "quintals/acre",
        "tons_per_acre": "tons/acre",
        "total_harvest": "Total harvest for",
        "acres": "acres",
        "quintals": "quintals",
        "model_details": "Model details",
        "model_info": "- **Model**: XGBoost | Features: N, P, K, pH, Rainfall\n- Prototype soil-response curves. Does not include pests/irrigation/variety.",
        "price_header": "Mandi Price Intelligence —",
        "in": "in",
        "no_data": "No recent data for",
        "try_other": "Try another combination.",
        "recent_avg": "Recent Avg Modal",
        "latest": "Latest Observed",
        "range": "Range",
        "records": "Records",
        "price_trend": "Recent Price Trend (₹/quintal)",
        "outlook": "14-day outlook: Peak ~₹",
        "trough": "| Trough ~₹",
        "top_markets": "Top Markets",
        "data_source": "Data source",
        "data_source_text": "Agmarknet historical mandi prices (cleaned). Modal ₹/quintal. Simple trend forecast.",
        "sell_header": "💡 Smart Selling Decision Advisor",
        "sell_intro": "Uses **real mandi prices** + transport cost to recommend where to sell.",
        "expected_yield": "Expected yield (quintals/acre)",
        "total_produce": "Total produce:",
        "insufficient": "Insufficient price data. Try another state/crop.",
        "local_option": "📍 Local Option",
        "higher_option": "🚛 Higher-Price Option",
        "transport": "Transport:",
        "net_revenue": "Net Revenue",
        "recommend_go": "**Recommendation: Go to",
        "extra": ".** Extra ~₹",
        "after_transport": " after transport.",
        "recommend_local": "**Recommendation: Sell locally (",
        "not_worth": ").** Distant market not worth the transport.",
        "prices_rising": "Prices rising. Consider holding 7–10 days if you have storage.",
        "prices_soft": "Prices softening. Selling sooner may reduce risk.",
        "assumptions": "Assumptions",
        "assumptions_text": "Real Agmarknet modal prices · Transport ₹{rate}/q/km · Distances approximate · Verify live rates on e-NAM/Agmarknet before moving produce.",
        "footer": "AgriTech SIH 2026 · Market linkages & price discovery · Agmarknet data",
        "weather_temp": "Temp",
        "weather_humidity": "Humidity",
        "weather_precip": "Precip",
        "weather_fail": "Weather fetch failed.",
        "lang_label": "Language / भाषा",
    },
    "hi": {
        "app_title": "🌾 एग्रीटेक: मूल्य पूर्वानुमान और उपज निर्णय इंजन",
        "app_sub": "SIH 2026 प्रोटोटाइप · फोकस: किसानों के लिए बाजार लिंकेज और मूल्य खोज",
        "sidebar_title": "📍 खेत की जानकारी",
        "state": "राज्य",
        "crop": "फसल",
        "farm_area": "खेत का क्षेत्रफल (एकड़)",
        "data_note": "डेटा: एगमार्कनेट 2023-2025 | मौसम: Open-Meteo",
        "tab_yield": "📊 उपज पूर्वानुमान",
        "tab_price": "📈 मंडी मूल्य जानकारी",
        "tab_sell": "💡 स्मार्ट बिक्री सुझाव",
        "yield_header": "फसल उपज का अनुमान लगाएं",
        "fetch_weather": "⛅ लाइव मौसम देखें",
        "soil_header": "मिट्टी और जलवायु जानकारी",
        "nitrogen": "नाइट्रोजन (N)",
        "phosphorus": "फास्फोरस (P)",
        "potassium": "पोटैशियम (K)",
        "ph": "मिट्टी का pH",
        "rainfall": "अपेक्षित वर्षा (मिमी)",
        "calc_yield": "उपज की गणना करें",
        "pred_yield": "अनुमानित उपज",
        "q_per_acre": "क्विंटल/एकड़",
        "tons_per_acre": "टन/एकड़",
        "total_harvest": "कुल पैदावार",
        "acres": "एकड़",
        "quintals": "क्विंटल",
        "model_details": "मॉडल विवरण",
        "model_info": "- **मॉडल**: XGBoost | विशेषताएँ: N, P, K, pH, वर्षा\n- यह एक प्रोटोटाइप है। कीट, सिंचाई या किस्म शामिल नहीं है।",
        "price_header": "मंडी मूल्य जानकारी —",
        "in": "में",
        "no_data": "इसके लिए हाल का डेटा नहीं मिला",
        "try_other": "दूसरा राज्य या फसल चुनें।",
        "recent_avg": "हाल का औसत भाव",
        "latest": "नवीनतम भाव",
        "range": "रेंज",
        "records": "रिकॉर्ड",
        "price_trend": "हाल का मूल्य रुझान (₹/क्विंटल)",
        "outlook": "14 दिन का अनुमान: अधिकतम ~₹",
        "trough": "| न्यूनतम ~₹",
        "top_markets": "मुख्य मंडियाँ",
        "data_source": "डेटा स्रोत",
        "data_source_text": "एगमार्कनेट ऐतिहासिक मंडी मूल्य (साफ़ किया हुआ)। मॉडल ₹/क्विंटल। सरल रुझान पूर्वानुमान।",
        "sell_header": "💡 स्मार्ट बिक्री निर्णय सलाहकार",
        "sell_intro": "वास्तविक मंडी भाव + परिवहन लागत के आधार पर बताता है कि कहाँ बेचना बेहतर है।",
        "expected_yield": "अपेक्षित उपज (क्विंटल/एकड़)",
        "total_produce": "कुल उपज:",
        "insufficient": "पर्याप्त मूल्य डेटा नहीं। दूसरा राज्य/फसल चुनें।",
        "local_option": "📍 स्थानीय विकल्प",
        "higher_option": "🚛 अधिक भाव वाला विकल्प",
        "transport": "परिवहन:",
        "net_revenue": "शुद्ध आय",
        "recommend_go": "**सुझाव: जाएँ",
        "extra": "।** परिवहन के बाद लगभग ₹",
        "after_transport": " अतिरिक्त मिल सकते हैं।",
        "recommend_local": "**सुझाव: स्थानीय मंडी में बेचें (",
        "not_worth": ")।** दूर की मंडी परिवहन के बाद फायदेमंद नहीं।",
        "prices_rising": "भाव बढ़ रहे हैं। यदि भंडारण है तो 7–10 दिन रुकने पर विचार करें।",
        "prices_soft": "भाव कमजोर हो रहे हैं। जल्दी बेचना जोखिम कम कर सकता है।",
        "assumptions": "मान्यताएँ",
        "assumptions_text": "वास्तविक एगमार्कनेट मॉडल भाव · परिवहन ₹{rate}/क्विंटल/किमी · दूरी अनुमानित · पैदावार ले जाने से पहले e-NAM/एगमार्कनेट पर लाइव भाव जाँचें।",
        "footer": "एग्रीटेक SIH 2026 · बाजार लिंकेज और मूल्य खोज · एगमार्कनेट डेटा",
        "weather_temp": "तापमान",
        "weather_humidity": "नमी",
        "weather_precip": "वर्षा",
        "weather_fail": "मौसम डेटा नहीं मिला।",
        "lang_label": "Language / भाषा",
    }
}

# =========================
# LOAD MODELS & DATA
# =========================
@st.cache_resource
def load_models():
    try:
        return joblib.load("yield_model.pkl")
    except Exception:
        return None

@st.cache_data
def load_price_data():
    try:
        df = pd.read_csv("price_data.csv")
        df["dt"] = pd.to_datetime(df["dt"])
        summary = pd.read_csv("price_summary_recent.csv")
        return df, summary
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

yield_model = load_models()
price_df, price_summary = load_price_data()

STATE_LIST = ["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Madhya Pradesh",
              "Rajasthan", "Gujarat", "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Bihar", "West Bengal"]
CROP_LIST = ["Wheat", "Rice", "Potato", "Tomato", "Onion"]
STATE_COORDS = {
    "Punjab": {"lat": 31.1471, "lon": 75.3412}, "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462}, "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Madhya Pradesh": {"lat": 23.2599, "lon": 77.4126}, "Rajasthan": {"lat": 26.9124, "lon": 75.7873},
    "Gujarat": {"lat": 23.2156, "lon": 72.6369}, "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569}, "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
    "Bihar": {"lat": 25.0961, "lon": 85.3131}, "West Bengal": {"lat": 22.9868, "lon": 87.8550}
}
TRANSPORT_RATE = 1.8

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
    market_stats = (recent.groupby("Market").agg(
        latest_price=("Modal_Price", "last"),
        avg_price=("Modal_Price", "mean"),
        n_days=("Modal_Price", "count"),
        last_date=("dt", "max")
    ).sort_values("n_days", ascending=False).head(6))
    return daily, market_stats.to_dict("index")

def simple_forecast(daily_series, horizon=14):
    if len(daily_series) < 7:
        last = daily_series["Modal_Price"].iloc[-1] if len(daily_series) > 0 else 2000
        return [last] * horizon, last, last
    y = daily_series["Modal_Price"].values[-30:]
    coef = np.polyfit(np.arange(len(y)), y, 1)
    last_price = y[-1]
    forecast = [max(last_price * 0.7, min(last_price * 1.4, last_price + coef[0] * (i + 1))) for i in range(horizon)]
    return forecast, max(forecast), min(forecast)

def get_summary_row(state, crop):
    if price_summary.empty:
        return None
    row = price_summary[(price_summary["State"] == state) & (price_summary["Commodity"] == crop)]
    return None if row.empty else row.iloc[0]

# =========================
# LANGUAGE TOGGLE
# =========================
if "lang" not in st.session_state:
    st.session_state.lang = "en"

lang_choice = st.sidebar.radio(
    LANG["en"]["lang_label"],
    options=["English", "हिंदी"],
    index=0 if st.session_state.lang == "en" else 1,
    horizontal=True
)
st.session_state.lang = "en" if lang_choice == "English" else "hi"
T = LANG[st.session_state.lang]

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.title(T["sidebar_title"])
state = st.sidebar.selectbox(T["state"], STATE_LIST, index=0)
crop = st.sidebar.selectbox(T["crop"], CROP_LIST, index=0)
farm_area = st.sidebar.number_input(T["farm_area"], min_value=0.5, value=5.0, step=0.5)
st.sidebar.markdown("---")
st.sidebar.caption(T["data_note"])

# =========================
# MAIN UI
# =========================
st.title(T["app_title"])
st.markdown(f"**{T['app_sub']}**")

tab1, tab2, tab3 = st.tabs([T["tab_yield"], T["tab_price"], T["tab_sell"]])

# ----- TAB 1: YIELD -----
with tab1:
    st.header(T["yield_header"])
    if st.button(f"{T['fetch_weather']} {state}"):
        coords = STATE_COORDS.get(state)
        if coords:
            try:
                url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,relative_humidity_2m,precipitation"
                r = requests.get(url, timeout=8).json().get("current", {})
                st.success(
                    f"**{state}** → {T['weather_temp']}: **{r.get('temperature_2m', 'N/A')}°C** | "
                    f"{T['weather_humidity']}: **{r.get('relative_humidity_2m', 'N/A')}%** | "
                    f"{T['weather_precip']}: **{r.get('precipitation', 'N/A')} mm**"
                )
            except Exception:
                st.warning(T["weather_fail"])

    st.subheader(T["soil_header"])
    c1, c2, c3 = st.columns(3)
    with c1:
        nitrogen = st.slider(T["nitrogen"], 0, 140, 70)
        phosphorus = st.slider(T["phosphorus"], 0, 145, 45)
    with c2:
        potassium = st.slider(T["potassium"], 0, 205, 35)
        ph_level = st.slider(T["ph"], 4.0, 9.0, 6.5, 0.1)
    with c3:
        rainfall = st.slider(T["rainfall"], 50, 400, 180)

    if st.button(T["calc_yield"], type="primary"):
        if yield_model is not None:
            pred_tons = max(0.3, round(float(yield_model.predict([[nitrogen, phosphorus, potassium, ph_level, rainfall]])[0]), 2))
        else:
            # fallback simple formula if model missing
            pred_tons = max(0.3, round(0.8 + nitrogen * 0.01 + phosphorus * 0.008 + potassium * 0.005 + (ph_level - 5.5) * 0.1 + rainfall * 0.002, 2))
        q_per_acre = round(pred_tons * 10, 1)
        st.success(f"**{T['pred_yield']}:** {q_per_acre} {T['q_per_acre']} ({pred_tons} {T['tons_per_acre']})")
        st.info(f"**{T['total_harvest']}** {farm_area} {T['acres']} → **{round(q_per_acre * farm_area, 1)} {T['quintals']}**")
        with st.expander(T["model_details"]):
            st.markdown(T["model_info"])

# ----- TAB 2: PRICE -----
with tab2:
    st.header(f"{T['price_header']} {crop} {T['in']} {state}")
    daily, market_stats = get_recent_prices(state, crop, 90)
    summary_row = get_summary_row(state, crop)

    if daily.empty and summary_row is None:
        st.warning(f"{T['no_data']} **{crop}** {T['in']} **{state}**. {T['try_other']}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        if summary_row is not None:
            m1.metric(T["recent_avg"], f"₹{int(summary_row['avg_modal'])}/q")
            m2.metric(T["latest"], f"₹{int(summary_row['latest_modal'])}/q")
            m3.metric(T["range"], f"₹{int(summary_row['min_p'])}–{int(summary_row['max_p'])}")
            m4.metric(T["records"], f"{int(summary_row['n_records'])}")

        if not daily.empty and len(daily) > 3:
            st.subheader(T["price_trend"])
            st.line_chart(daily.set_index("Date"), height=280)
            _, peak, trough = simple_forecast(daily)
            st.caption(f"{T['outlook']}{int(peak)} {T['trough']}{int(trough)}")

        if market_stats:
            st.subheader(T["top_markets"])
            rows = [{
                "Market": m,
                "Latest Modal ₹": int(s["latest_price"]),
                "Avg ₹": int(s["avg_price"]),
                "Days": int(s["n_days"]),
                "Last Date": str(s["last_date"].date())
            } for m, s in market_stats.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander(T["data_source"]):
            st.markdown(T["data_source_text"])

# ----- TAB 3: SMART SELL -----
with tab3:
    st.header(T["sell_header"])
    st.markdown(T["sell_intro"])
    daily, market_stats = get_recent_prices(state, crop, 60)
    summary_row = get_summary_row(state, crop)

    default_q = 18.0 if crop == "Wheat" else (12.0 if crop in ["Potato", "Onion"] else 15.0)
    expected_q = st.number_input(T["expected_yield"], min_value=1.0, value=default_q, step=0.5)
    total_q = expected_q * farm_area
    st.write(f"**{T['total_produce']}** **{total_q:.1f} {T['quintals']}**")

    if not market_stats and summary_row is None:
        st.warning(T["insufficient"])
    else:
        if market_stats:
            ms = sorted(market_stats.items(), key=lambda x: x[1]["latest_price"], reverse=True)
            local_mkt, local_s = ms[-1] if len(ms) > 1 else ms[0]
            best_mkt, best_s = ms[0]
            price_local, price_best = int(local_s["latest_price"]), int(best_s["latest_price"])
            dist_local, dist_best = 12, 45
        else:
            price_local = int(summary_row["median_modal"])
            price_best = int(summary_row["max_p"] * 0.92)
            local_mkt, best_mkt = f"{state} local", f"{state} higher-price"
            dist_local, dist_best = 10, 40

        cost_l = TRANSPORT_RATE * dist_local * total_q
        cost_b = TRANSPORT_RATE * dist_best * total_q
        net_l = price_local * total_q - cost_l
        net_b = price_best * total_q - cost_b
        delta = net_b - net_l

        ca, cb = st.columns(2)
        with ca:
            st.subheader(T["local_option"])
            st.write(f"**{local_mkt}** · ~{dist_local} km · ₹{price_local}/q")
            st.write(f"{T['transport']} ₹{cost_l:,.0f}")
            st.metric(T["net_revenue"], f"₹{net_l:,.0f}")
        with cb:
            st.subheader(T["higher_option"])
            st.write(f"**{best_mkt}** · ~{dist_best} km · ₹{price_best}/q")
            st.write(f"{T['transport']} ₹{cost_b:,.0f}")
            st.metric(T["net_revenue"], f"₹{net_b:,.0f}", delta=f"₹{delta:,.0f}")

        st.markdown("---")
        if net_b > net_l * 1.03:
            st.success(f"{T['recommend_go']} {best_mkt}{T['extra']}{delta:,.0f}{T['after_transport']}")
        else:
            st.info(f"{T['recommend_local']}{local_mkt}{T['not_worth']}")

        if not daily.empty and len(daily) >= 10:
            trend = daily["Modal_Price"].iloc[-5:].mean() - daily["Modal_Price"].iloc[-15:-5].mean()
            if trend > 30:
                st.warning(T["prices_rising"])
            elif trend < -40:
                st.warning(T["prices_soft"])

        with st.expander(T["assumptions"]):
            st.markdown(T["assumptions_text"].format(rate=TRANSPORT_RATE))

st.markdown("---")
st.caption(T["footer"])
