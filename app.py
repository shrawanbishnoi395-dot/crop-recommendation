*(Make sure there is a hyphen `-` in `scikit-learn`, not an underscore, and no spaces).*
4. Scroll down and click the green **"Commit changes"** button.

---

### Step 2: Fix the Code in `app.py`

The error on line 47 happens because your `app.py` script uses the old library name (`pickle`) instead of loading it through `joblib` or the backend configuration is pointing to an un-imported module.

1. Go back to your main repository page on GitHub.
2. Click on **`app.py`**, click the **pencil icon** to edit.
3. Completely replace all the text inside with this clean, updated deployment version:

```python
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load the exported deployment files safely
try:
    with open('best_crop_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('feature_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'N_P_Ratio', 'Climate_Strain']
except Exception as e:
    st.error(f"Initialization Error: Core assets missing or corrupted. Details: {e}")

# 2. User Interface Configuration
st.set_page_config(page_title="AI Agriculture System", page_icon="🌱", layout="centered")

st.title("🌱 Intelligent Crop Recommendation System")
st.write("Enter the soil nutrients and climate factors below to find the most suitable crop for cultivation.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Soil Nutrients")
    N = st.slider("Nitrogen (N) Content", min_value=0, max_value=150, value=50)
    P = st.slider("Phosphorus (P) Content", min_value=5, max_value=150, value=50)
    K = st.slider("Potassium (K) Content", min_value=5, max_value=210, value=50)
    ph = st.slider("Soil pH Level", min_value=3.5, max_value=10.0, value=6.5, step=0.1)

with col2:
    st.subheader("🌤️ Environmental Factors")
    temp = st.slider("Temperature (°C)", min_value=8.0, max_value=50.0, value=25.0, step=0.5)
    humidity = st.slider("Relative Humidity (%)", min_value=14.0, max_value=100.0, value=70.0, step=0.5)
    rainfall = st.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0, step=1.0)

st.markdown("---")

# 3. Execution Pipeline Action
if st.button("🚀 Recommend Optimal Crop", use_container_width=True):
    with st.spinner("Analyzing parameters..."):
        # Explicit math conversions to avoid missing attribute structures
        n_p_ratio = float(N) / (float(P) + 1e-5)
        climate_strain = float(temp) * float(humidity)
        
        input_data = pd.DataFrame([[float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall), n_p_ratio, climate_strain]], 
                                  columns=feature_columns)
        
        try:
            input_scaled = scaler.transform(input_data)
            recommended_crop = model.predict(input_scaled)[0]
            st.balloons()
            st.success(f"### 🎉 Recommended Crop: **{str(recommended_crop).upper()}**")
        except Exception as eval_error:
            st.error(f"Execution Error during model evaluation: {eval_error}")
