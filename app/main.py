# app/main.py

import streamlit as st
import pickle
import pandas as pd
import os

# ------------------------------
# Load models
# ------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

model_files = {
    "Linear Regression": "linear.pkl",
    "Ridge": "ridge.pkl",
    "Lasso": "lasso.pkl",
    "Random Forest": "rf.pkl",
    "KNN": "knn.pkl"
}

models = {}
for name, file in model_files.items():
    path = os.path.join(MODEL_DIR, file)
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

# ------------------------------
# Define features with min/max from dataset
# ------------------------------

features_info = {
    "Hour": (0, 23, 12),
    "Temperature(°C)": (-17.8, 39.4, 20),
    "Humidity(%)": (0, 98, 50),
    "Wind speed (m/s)": (0, 7.4, 2),
    "Visibility (10m)": (27, 2000, 1000),
    "Dew point temperature(°C)": (-30.6, 27.2, 0),
    "Solar Radiation (MJ/m2)": (0, 3.52, 1),
    "Rainfall(mm)": (0, 35, 0),
    "Snowfall (cm)": (0, 8.8, 0),
    "Holiday": (0, 1, 0),
    "Functioning Day": (0, 1, 1),
    "Month": (1, 12, 6)
}

st.title("Seoul Bike Demand Prediction")
st.write("Adjust all features to predict bike rentals:")

# ------------------------------
# User inputs
# ------------------------------

user_inputs = {}
for feature, (min_val, max_val, default) in features_info.items():
    if feature in ["Holiday", "Functioning Day", "Month", "Hour"]:
        # discrete selection
        user_inputs[feature] = st.selectbox(
            feature,
            options=list(range(int(min_val), int(max_val)+1)),
            index=int(default-min_val)
        )
    else:
        # continuous slider
        user_inputs[feature] = st.slider(feature, float(min_val), float(max_val), float(default))

# Convert to DataFrame
input_df = pd.DataFrame([user_inputs], columns=list(features_info.keys()))

st.subheader("User Inputs")
st.dataframe(input_df)

# ------------------------------
# Predictions
# ------------------------------

st.subheader("Predictions")
for name, model in models.items():
    try:
        prediction = model.predict(input_df)[0]
        st.write(f"**{name}:** {prediction:.0f} bikes")
    except Exception as e:
        st.write(f"**{name}:** Error - {e}")
