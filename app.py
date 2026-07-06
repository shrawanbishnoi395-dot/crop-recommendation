import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model_and_scaler():
    try:
        with open("best_crop_model.pkl", "rb") as model_file:
            model = joblib.load(model_file)

        with open("feature_scaler.pkl", "rb") as scaler_file:
            scaler = joblib.load(scaler_file)

    except:
        with open("best_crop_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

        with open("feature_scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

    return model, scaler


try:
    model, scaler = load_model_and_scaler()

except Exception as e:
    st.error(f"Error Loading Model : {e}")
    st.stop()

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🌱 Crop Recommendation")

st.sidebar.markdown(
"""
### About

This application predicts the most suitable crop based on soil nutrients and environmental conditions.

### Tech Stack

- Python
- Scikit-learn
- Streamlit
- Random Forest

---
Developed as an ML Portfolio Project.
"""
)

# ---------------- TITLE ---------------- #

st.title("🌱 Intelligent Crop Recommendation System")

st.write(
"Predict the most suitable crop using soil nutrients and environmental conditions."
)

st.markdown("---")

# ---------------- METRICS ---------------- #

c1, c2, c3, c4 = st.columns(4)

c1.metric("🤖 Model", "Random Forest")
c2.metric("🎯 Accuracy", "99.4%")
c3.metric("📊 Features", "7")
c4.metric("🌾 Crops", "22")

st.markdown("---")

# ---------------- INPUT ---------------- #

left, right = st.columns(2)

with left:

    st.subheader("🧪 Soil")

    Nitrogen = st.slider("Nitrogen", 0, 150, 50)

    Phosphorus = st.slider("Phosphorus", 5, 150, 50)

    Potassium = st.slider("Potassium", 5, 210, 50)

    pH_Value = st.slider(
        "pH",
        3.5,
        10.0,
        6.5,
        0.1
    )

with right:

    st.subheader("🌦 Climate")

    Temperature = st.slider(
        "Temperature (°C)",
        8.0,
        50.0,
        25.0,
        0.5
    )

    Humidity = st.slider(
        "Humidity (%)",
        14.0,
        100.0,
        70.0,
        0.5
    )

    Rainfall = st.slider(
        "Rainfall (mm)",
        20.0,
        300.0,
        100.0,
        1.0
    )

st.markdown("---")

# ---------------- PREDICTION ---------------- #

if st.button("🚀 Recommend Crop", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            float(Nitrogen),
            float(Phosphorus),
            float(Potassium),
            float(Temperature),
            float(Humidity),
            float(pH_Value),
            float(Rainfall)
        ]],
        columns=[
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Temperature",
            "Humidity",
            "pH_Value",
            "Rainfall"
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    best_index = np.argmax(probability)

    confidence = probability[best_index] * 100

    st.success(f"## 🌾 Recommended Crop\n### {prediction.upper()}")

    st.progress(confidence / 100)

    st.metric("Confidence Score", f"{confidence:.2f}%")

    st.markdown("### 🏆 Top 3 Recommendations")

    top3 = np.argsort(probability)[::-1][:3]

    medals = ["🥇", "🥈", "🥉"]

    for medal, idx in zip(medals, top3):

        st.write(
            f"{medal} **{model.classes_[idx]}** — {probability[idx]*100:.2f}%"
        )

   with st.expander("📋 Input Summary", expanded=True):

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Nitrogen", Nitrogen)
        st.metric("Phosphorus", Phosphorus)
        st.metric("Potassium", Potassium)
        st.metric("pH", pH_Value)

    with col2:
        st.metric("Temperature", f"{Temperature} °C")
        st.metric("Humidity", f"{Humidity}%")
        st.metric("Rainfall", f"{Rainfall} mm")

    st.balloons()

st.markdown("---")

st.caption(
    "Made with ❤️ using Python • Scikit-learn • Streamlit"
)
