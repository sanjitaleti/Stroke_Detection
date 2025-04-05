import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained models and scaler
clf = joblib.load("models/stroke_risk_classifier.pkl")
reg = joblib.load("models/stroke_risk_regressor.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define feature names
feature_names = [
    "Chest Pain", "Shortness of Breath", "Irregular Heartbeat", "Fatigue & Weakness",
    "Dizziness", "Swelling (Edema)", "Pain in Neck/Jaw/Shoulder/Back", "Excessive Sweating",
    "Persistent Cough", "Nausea/Vomiting", "High Blood Pressure", "Chest Discomfort (Activity)",
    "Cold Hands/Feet", "Snoring/Sleep Apnea", "Anxiety/Feeling of Doom", "Age"
]

# Apply custom CSS for styling
st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
        .stApp { background-color: #1E1E1E; }
        .main-title { text-align: center; font-size: 36px; font-weight: bold; color: #FF4B4B; }
        .sub-header { font-size: 24px; color: #FFFFFF; }
        .stButton>button { background-color: #FF4B4B; color: white; font-weight: bold; border-radius: 8px; padding: 10px; }
        .stButton>button:hover { background-color: #E63946; }
        .risk-box { font-size: 24px; font-weight: bold; text-align: center; padding: 10px; border-radius: 10px; margin-top: 10px; }
        .low-risk { background-color: #4CAF50; color: white; }
        .high-risk { background-color: #FF4B4B; color: white; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Stroke_illustration_%28CDC%29.png", use_container_width=True)
st.sidebar.title("⚕️ Stroke Risk Prediction App")

st.sidebar.markdown("""
### 📌 About This Project  
Stroke is a **leading cause of disability and death worldwide**.  
This AI-powered tool predicts **stroke risk** based on patient symptoms and medical history.

### 🔬 How It Works  
- Uses **machine learning** to assess risk factors  
- Provides a **binary classification** (At Risk or Not)  
- Estimates **Stroke Risk Percentage (%)**  
- Helps in **early detection and prevention**  

### 🚀 Get Started  
Enter patient symptoms to receive an instant **stroke risk assessment**.
""")

# Main Title
st.markdown("<h1 style='text-align: center; font-size: 3rem; color: #ff4b4b;'>🧠 Stroke Risk Prediction</h1>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://www.cdc.gov/stroke/images/stroke-brain.png", use_container_width=True)

with col2:
    st.markdown('<p class="sub-header">📋 Enter Patient Details</p>', unsafe_allow_html=True)

    # Create input fields
    user_input = {}
    for feature in feature_names:
        if feature == "Age":
            user_input[feature] = st.slider(f"{feature} (years)", 0, 120, 30, format="%d")  # Fix for extra 0 issue
        else:
            user_input[feature] = st.toggle(f"{feature}")

# Prediction button
if st.button("🚑 Predict Stroke Risk"):
    input_df = pd.DataFrame([{feature: int(user_input[feature]) for feature in feature_names}])
    input_scaled = scaler.transform(input_df)

    # Get predictions
    at_risk = clf.predict(input_scaled)[0]
    stroke_risk_percentage = reg.predict(input_scaled)[0]

    # Display Results
    st.markdown('<p class="sub-header">📊 Prediction Results</p>', unsafe_allow_html=True)

    # Colorful Result Box
    risk_class = "high-risk" if at_risk == 1 else "low-risk"
    st.markdown(f'<div class="risk-box {risk_class}">'
                f'At Risk: {"Yes 🚨" if at_risk == 1 else "No ✅"}</div>', 
                unsafe_allow_html=True)

    # Show Stroke Risk Percentage with Animated Progress Bar
    st.write(f"**Estimated Stroke Risk:** {stroke_risk_percentage:.1f}%")  # Fixed formatting
    st.progress(min(int(stroke_risk_percentage), 100))

