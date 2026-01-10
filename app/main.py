# app/main.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Paths and load models
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
# Features info (min, max, default)
# ------------------------------
features_info = {
    "Hour": (0, 23, 12),
    "Month": (1, 12, 6),
    "Holiday": (0, 1, 0),
    "Functioning Day": (0, 1, 1),
    "Temperature(°C)": (-17.8, 39.4, 20),
    "Dew point temperature(°C)": (-30.6, 27.2, 0),
    "Humidity(%)": (0, 98, 50),
    "Wind speed (m/s)": (0, 7.4, 2),
    "Visibility (10m)": (27, 2000, 1000),
    "Solar Radiation (MJ/m2)": (0, 3.52, 1),
    "Rainfall(mm)": (0, 35, 0),
    "Snowfall (cm)": (0, 8.8, 0)
}

feature_order = [
    "Hour", "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)",
    "Dew point temperature(°C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)",
    "Snowfall (cm)", "Holiday", "Functioning Day", "Month"
]

numeric_features = ["Temperature(°C)", "Dew point temperature(°C)", "Humidity(%)",
                    "Wind speed (m/s)", "Visibility (10m)", "Solar Radiation (MJ/m2)",
                    "Rainfall(mm)", "Snowfall (cm)"]

# ------------------------------
# Sidebar inputs
# ------------------------------
st.sidebar.title("Bike Demand Input")

selected_model = st.sidebar.selectbox("Select Model", list(models.keys()), index=3)  # RF default

user_inputs = {}
for feature in ["Hour", "Month", "Holiday", "Functioning Day"]:
    min_val, max_val, default = features_info[feature]
    user_inputs[feature] = st.sidebar.selectbox(
        feature,
        options=list(range(int(min_val), int(max_val)+1)),
        index=int(default - min_val)
    )

for feature in numeric_features:
    min_val, max_val, default = features_info[feature]
    user_inputs[feature] = st.sidebar.slider(feature, float(min_val), float(max_val), float(default))

apply_button = st.sidebar.button("Apply Changes")

# ------------------------------
# Main frame
# ------------------------------
if apply_button:
    input_df = pd.DataFrame([user_inputs])
    input_df = input_df.reindex(columns=feature_order)

    # Predict selected model
    try:
        model = models[selected_model]
        pred = model.predict(input_df)[0]
        st.subheader(f"Prediction ({selected_model})")
        st.metric(label="Predicted Bike Count", value=int(pred))

        # Recommendation
        if pred < 500:
            rec = "Low usage 🚴"
        elif pred < 1500:
            rec = "Moderate usage 🚴‍♂️"
        else:
            rec = "High usage 🚴‍♀️"
        st.write(f"**Recommendation:** {rec}")

    except Exception as e:
        st.error(f"Error predicting: {e}")

    # Compare RF vs selected model
    if selected_model != "Random Forest":
        rf_pred = models["Random Forest"].predict(input_df)[0]
        st.subheader("Comparison with Random Forest")
        fig, ax = plt.subplots()
        sns.barplot(x=[selected_model, "Random Forest"], y=[pred, rf_pred], palette="magma", ax=ax)
        ax.set_ylabel("Predicted Bike Count")
        st.pyplot(fig)


    # Feature importance (RF only)
    st.subheader("Random Forest Feature Importance")

    rf_model = models["Random Forest"]

    # Check if it's a pipeline
    if hasattr(rf_model, "named_steps"):
        # If pipeline, get the RandomForestRegressor inside
        rf_model_raw = None
        for step in rf_model.named_steps.values():
            if hasattr(step, "feature_importances_"):
                rf_model_raw = step
                break
        if rf_model_raw is None:
            st.error("Random Forest regressor not found in pipeline.")
    else:
        rf_model_raw = rf_model  # plain RF model

    # Plot feature importance
    if hasattr(rf_model_raw, "feature_importances_"):
        fi = rf_model_raw.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_order, "Importance": fi})
        fi_df = fi_df.sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax)
        st.pyplot(fig)
    else:
        st.error("Feature importances not available for Random Forest model.")
