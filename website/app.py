import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Home Energy Consumption Optimization", layout="wide")
st.markdown("<h1 style='text-align: center;'>Smart Home Energy Consumption Optimization</h1>", unsafe_allow_html=True)

# --- Load Models ---
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"❌ Model not found at: {path}")
        return None

isol_model = load_model("models/isol.joblib")
xgb_reg = load_model("models/Xgboost.joblib")
xgb_cls = load_model("models/Xgboost_C.joblib")
rf_cls = load_model("models/Ran_for_C.joblib")

# --- Input Form ---
with st.form("input_form"):
    st.subheader("📋 Enter Smart Home Electricity Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.slider("🌡️ Temperature (°C)", 10.0, 45.0, 30.0)
        humidity = st.slider("💦 Humidity (%)", 10.0, 100.0, 70.0)
        occupancy = st.number_input("👥 Occupancy", min_value=0, value=3)

    with col2:
        appliances = st.number_input("🔌 Appliances Used", min_value=0, value=5)
        total_energy = st.number_input("⚡ Total Energy (kWh)", min_value=0.0, value=15.0)
        tariff = st.number_input("₹ Tariff per kWh", min_value=0.0, value=6.5)

    with col3:
        bill = st.number_input("🧾 Total Bill (₹)", min_value=0.0, value=97.5)
        day = st.selectbox("📅 Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        weather = st.selectbox("☁️ Weather", ["sunny", "cloudy", "rainy", "windy"])

    hour = st.slider("⏰ Hour", 0, 23, 14)
    minute = st.slider("🕒 Minute", 0, 59, 0)

    submit = st.form_submit_button("🚀 Run Prediction")

# --- Model Predictions ---
if submit:
    st.markdown("---")
    st.subheader("📈 Model Predictions & Visual Insights")

    input_data1 = {
        "temperature": temperature,
        "humidity": humidity,
        "occupancy": occupancy,
        "number_of_appliances_used": appliances,
        "total_energy_kwh": total_energy,
        "tariff_rs_per_kwh": tariff,
        "total_electricity_bill_rs": bill
    }

    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        input_data1[f"day_of_week_{d}"] = 1.0 if day == d else 0.0

    for w in ["sunny", "cloudy", "rainy", "windy"]:
        input_data1[f"weather_{w}"] = 1.0 if weather == w else 0.0

    input_data1['hour'] = hour
    input_data1['minute'] = minute

    month = datetime.now().month
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

    input_reg = pd.DataFrame([input_data1])
    input_data2 = {
        "temperature": temperature,
        "humidity": humidity,
        "occupancy": occupancy,
        "number_of_appliances_used": appliances,
        "total_energy_kwh": total_energy,
        "tariff_rs_per_kwh": tariff,
        "total_electricity_bill_rs": bill
    }

    input_cls = input_data2.copy()

    input_cls["month"] = month
    input_cls['hour'] = hour
    input_cls["is_weekend"] = is_weekend

    for w in [ 'cloudy',"rainy", "sunny", "windy"]:
        input_cls[f"weather_{w}"] = (weather == w)

    for d in [ 'Friday',"Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]:
        input_cls[f"day_of_week_{d}"] = (day == d)

    input_cls_df = pd.DataFrame([input_cls])

    input_iso = input_data1.copy()
    for col in ["minute", "tariff_rs_per_kwh", "total_electricity_bill_rs"]:
        input_iso.pop(col, None)

    input_iso_df = pd.DataFrame([input_iso])

    col1, col2 = st.columns(2)

    # 🔋 Regression: XGBoost
    with col1:
        st.markdown("### 🔋 Predicted Electricity Consumption (kWh)")
        if xgb_reg:
            try:
                input_reg = input_reg[xgb_reg.feature_names_in_]
                y_pred = xgb_reg.predict(input_reg)[0]
                st.success(f"📊 Predicted: **{y_pred:.2f} kWh**")
                fig = px.bar(x=["Predicted Consumption"], y=[y_pred], color=["kWh"],
                             labels={"x": "Type", "y": "kWh"}, height=300)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Regression Error: {e}")

    # 🏷️ Classification: XGBoost
    with col2:
        st.markdown("### 🏷️ XGBoost Classification Output")
        if xgb_cls:
            try:
                input_cls_df = input_cls_df[xgb_cls.feature_names_in_]
                cls_pred = xgb_cls.predict(input_cls_df)[0]
                st.info(f"📌 Category: **{cls_pred}**")
                fig2 = px.pie(values=[1], names=[cls_pred],
                              title="XGBoost Consumption Category")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ XGBoost Classifier Error: {e}")

    col3, col4 = st.columns(2)

    # 🌲 Classification: Random Forest
    with col3:
        st.markdown("### 🌲 Random Forest Classification Output")
        if rf_cls:
            try:
                input_cls_df = input_cls_df[rf_cls.feature_names_in_]
                cls_pred2 = rf_cls.predict(input_cls_df)[0]
                st.info(f"📌 Category: **{cls_pred2}**")
                fig3 = px.bar(x=["Random Forest"], y=[1], color=[cls_pred2],
                              labels={"x": "Model", "y": "Count"},
                              title="Random Forest Prediction")
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Random Forest Classifier Error: {e}")

    # 🚨 Isolation Forest
    with col4:
        st.markdown("### 🚨 Anomaly Detection (Isolation Forest)")
        if isol_model:
            try:
                input_iso_df = input_iso_df[isol_model.feature_names_in_]
                anomaly = isol_model.predict(input_iso_df)[0]
                if anomaly == -1:
                    st.error("🔴 Anomaly Detected!")
                    values = [1, 0]
                    labels = ["Anomaly", "Normal"]
                    colors = ["red", "green"]
                else:
                    st.success("🟢 Normal Usage")
                    values = [0, 1]
                    labels = ["Anomaly", "Normal"]
                    colors = ["red", "green"]

                fig4 = px.bar(x=labels, y=values, color=labels,
                              color_discrete_sequence=colors,
                              labels={"x": "Status", "y": "Count"},
                              title="Anomaly Detection Result")
                st.plotly_chart(fig4, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Isolation Forest Error: {e}")

    # --- Summary & Suggestions ---
    st.markdown("---")
    st.markdown("## 📘 Summary & Suggestions")

    with st.expander("🔍 View Your Input Summary"):
        st.write(pd.DataFrame([input_data1]))

    st.success("""
🎯 **Summary**:
- The system has predicted your electricity consumption and classified your usage pattern using:
  - XGBoost Regression
  - XGBoost & Random Forest Classifiers
  - Isolation Forest for anomaly detection

✅ Predictions are visualized with interactive graphs.

💡 **Suggestions**:
- Reduce unnecessary appliance usage to lower peak consumption.
- Monitor weather & occupancy impact on electricity bills.
- Track anomalies to detect sudden spikes or device faults.
- Use predictions to plan better energy usage and save cost.
""")

    st.info("📩 Have feedback or ideas to improve? Let us know!")
