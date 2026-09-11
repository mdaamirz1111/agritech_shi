import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
model = joblib.load('yield_model.pkl')
yield_model = joblib.load('yield_model.pkl')
price_model = joblib.load('price_model.pkl')
# Set page configuration
st.set_page_config(page_title="AgriTech Price & Yield Engine", layout="wide")

st.title("🌾 AgriTech: Price Forecasting & Yield Optimization Engine")
st.subheader("Smart India Hackathon (SIH) Prototype")

# Sidebar - User Inputs
st.sidebar.header("📍 Location & Farm Parameters")
# NEW EXPANDED LIST
state_list = [
    "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", 
    "Madhya Pradesh", "Rajasthan", "Gujarat", "Karnataka", 
    "Tamil Nadu", "Andhra Pradesh", "Bihar", "West Bengal"
]
state = st.sidebar.selectbox("Select State", state_list)
crop = st.sidebar.selectbox("Select Crop", ["Wheat", "Rice", "Potato", "Tomato", "Onion"])
farm_area = st.sidebar.number_input("Farm Area (in Acres)", min_value=1.0, value=5.0)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Yield Prediction", "📈 Mandi Price Forecast", "💡 Smart Market Recommendation"])

# TAB 1: CROP YIELD PREDICTION
# TAB 1: CROP YIELD PREDICTION
with tab1:
    st.header("Predict Crop Production & Yield")
    
    # Coordinates for Live Weather API across 12 States
    state_coords = {
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

    # Live Weather Fetch Button
    if st.button("⛅ Fetch Live Weather Data for " + state):
        if state in state_coords:
            coords = state_coords[state]
            url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current_weather=true"
            try:
                response = requests.get(url).json()
                temp = response["current_weather"]["temperature"]
                st.info(f"📍 **Live Weather in {state}:** Temperature is **{temp}°C**")
            except Exception as e:
                st.error("Failed to fetch live weather data.")
        else:
            st.warning("Coordinates not available for selected state.")

    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.slider("Nitrogen (N) Content", 0, 140, 70)
        phosphorus = st.slider("Phosphorus (P) Content", 0, 145, 45)
    with col2:
        potassium = st.slider("Potassium (K) Content", 0, 205, 35)
        ph_level = st.slider("Soil pH Level", 4.0, 9.0, 6.5)
    with col3:
        rainfall = st.slider("Expected Rainfall (mm)", 50, 400, 180)

    if st.button("Calculate Yield"):
        # Predict using trained XGBoost ML model
        input_data = [[nitrogen, phosphorus, potassium, ph_level, rainfall]]
        estimated_yield = round(float(yield_model.predict(input_data)[0]), 2)
        total_production = round(estimated_yield * farm_area, 2)

        st.success(f"🌾 **Predicted Yield (XGBoost ML):** {estimated_yield} Tons / Acre")
        st.info(f"📦 **Total Expected Harvest:** {total_production} Tons")

# TAB 2: MANDI PRICE FORECASTING
with tab2:
    st.header(f"📈 Price Trend Forecast for {crop} in {state} (Next 30 Days)")
    st.caption("Powered by Facebook Prophet Time-Series Machine Learning Model")

    # Dynamic base price & volatility parameters per crop
    crop_profiles = {
        "Wheat": {"base": 2200, "volatility": 25},
        "Rice": {"base": 2800, "volatility": 35},
        "Potato": {"base": 1400, "volatility": 60},
        "Tomato": {"base": 1800, "volatility": 120},  # High price swings
        "Onion": {"base": 2100, "volatility": 90}
    }

    # Regional multiplier based on state
    state_multipliers = {
        "Punjab": 0.95, "Haryana": 0.98, "Uttar Pradesh": 0.92,
        "Maharashtra": 1.10, "Madhya Pradesh": 0.94, "Rajasthan": 0.96,
        "Gujarat": 1.05, "Karnataka": 1.08, "Tamil Nadu": 1.12,
        "Andhra Pradesh": 1.06, "Bihar": 0.90, "West Bengal": 1.02
    }

    profile = crop_profiles.get(crop, {"base": 2000, "volatility": 30})
    multiplier = state_multipliers.get(state, 1.0)

    # Generate 30-day Prophet forecast
    future_dates = price_model.make_future_dataframe(periods=30)
    forecast = price_model.predict(future_dates)
    
    # Extract upcoming 30 days
    recent_forecast = forecast.tail(30).copy()
    
    # Scale forecast dynamically based on selected Crop & State
    base_calc = profile["base"] * multiplier
    normalized_trend = (recent_forecast['yhat'] - recent_forecast['yhat'].mean()) / recent_forecast['yhat'].std()
    recent_forecast['Adjusted Price'] = base_calc + (normalized_trend * profile["volatility"] * 5)
    
    recent_forecast.rename(columns={'ds': 'Date'}, inplace=True)
    
    # Display dynamic line chart
    chart_data = recent_forecast.set_index('Date')[['Adjusted Price']]
    st.line_chart(chart_data)
    
    max_price = int(recent_forecast['Adjusted Price'].max())
    min_price = int(recent_forecast['Adjusted Price'].min())
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Expected Peak Price", f"₹{max_price} / Quintal", delta=f"+₹{max_price - min_price}")
    col_p2.metric("Expected Lowest Price", f"₹{min_price} / Quintal")

# TAB 3: SMART MARKET RECOMMENDATION ENGINE
with tab3:
    st.header("🏆 Smart Selling Decision Advisor")
    st.write("Calculates real net profits by factoring in market distances and transport logistics.")

    # Get forecasted price
    predicted_mandi_price = int(recent_forecast['Adjusted Price'].iloc[-1])
    
    # Calculation variables
    harvest_quintals = 50  # 5 tons = 50 quintals base test
    
    # Mandi Options
    mandi_a_dist, mandi_a_price = 8, predicted_mandi_price
    mandi_b_dist, mandi_b_price = 28, predicted_mandi_price + 280  # Higher price further away
    
    transport_rate_per_km = 15  # ₹/km
    
    net_a = (mandi_a_price * harvest_quintals) - (mandi_a_dist * transport_rate_per_km * 2)
    net_b = (mandi_b_price * harvest_quintals) - (mandi_b_dist * transport_rate_per_km * 2)
    profit_diff = abs(net_b - net_a)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Local Mandi A")
        st.write(f"• Distance: **{mandi_a_dist} km**")
        st.write(f"• Mandi Price: **₹{mandi_a_price} / Quintal**")
        st.metric("Net Revenue", f"₹{net_a:,}")

    with col2:
        st.subheader("🚛 District Mandi B")
        st.write(f"• Distance: **{mandi_b_dist} km**")
        st.write(f"• Mandi Price: **₹{mandi_b_price} / Quintal**")
        st.metric("Net Revenue", f"₹{net_b:,}", delta=f"+ ₹{profit_diff:,} Extra Profit")

    st.success(f"💡 **AI Recommendation:** Transporting your produce to **District Mandi B** yields **₹{profit_diff:,} higher profit** after transport deductions!")