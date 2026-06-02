import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load the exported deployment files safely
@st.cache_resource
def load_model_and_scaler():
    try:
        # Try joblib first (more robust for sklearn objects)
        with open('best_crop_model.pkl', 'rb') as model_file:
            model = joblib.load(model_file)
        with open('feature_scaler.pkl', 'rb') as scaler_file:
            scaler = joblib.load(scaler_file)
    except:
        # Fallback to pickle
        with open('best_crop_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('feature_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
    # Get the actual feature names from the scaler
    if hasattr(scaler, 'get_feature_names_out'):
        feature_columns = list(scaler.get_feature_names_out())
    elif hasattr(scaler, 'feature_names_in_'):
        feature_columns = list(scaler.feature_names_in_)
    else:
        # Fallback: Use the features the scaler was fitted on
        feature_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
except Exception as e:
    st.error(f"❌ Initialization Error: Core assets missing or corrupted. Details: {e}")
    st.stop()

# User Interface Configuration
st.set_page_config(page_title="AI Agriculture System", page_icon="🌱", layout="centered")

st.title("🌱 Intelligent Crop Recommendation System")
st.write("Enter the soil nutrients and climate factors below to find the most suitable crop for cultivation.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Soil Nutrients")
    Nitrogen = st.slider("Nitrogen (N) Content", min_value=0, max_value=150, value=50)
    Phosphorus = st.slider("Phosphorus (P) Content", min_value=5, max_value=150, value=50)
    Potassium = st.slider("Potassium (K) Content", min_value=5, max_value=210, value=50)
    pH_Value = st.slider("Soil pH Level", min_value=3.5, max_value=10.0, value=6.5, step=0.1)

with col2:
    st.subheader("🌤️ Environmental Factors")
    Temperature = st.slider("Temperature (°C)", min_value=8.0, max_value=50.0, value=25.0, step=0.5)
    Humidity = st.slider("Relative Humidity (%)", min_value=14.0, max_value=100.0, value=70.0, step=0.5)
    Rainfall = st.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0, step=1.0)

st.markdown("---")

# Execution Pipeline Action
if st.button("🚀 Recommend Optimal Crop", use_container_width=True):
    with st.spinner("Analyzing parameters..."):
        try:
            # Create input data with only the base features (no engineered features yet)
            input_data = pd.DataFrame(
                [[float(Nitrogen), float(Phosphorus), float(Potassium), float(Temperature), float(Humidity), float(pH_Value), float(Rainfall)]], 
                columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
            )
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Predict result
            recommended_crop = model.predict(input_scaled)[0]
            st.balloons()
            st.success(f"### 🎉 Recommended Crop: **{str(recommended_crop).upper()}**")
        except Exception as eval_error:
            st.error(f"❌ Execution Error during model evaluation: {eval_error}")
            st.write(f"Expected features: {feature_columns}")
