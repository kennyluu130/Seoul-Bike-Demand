# app/main.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

### Paths and load models ###

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

model_files = {
    "Linear Regression": "linear.pkl",
    "Ridge": "ridge.pkl",
    "Lasso": "lasso.pkl",
    "Random Forest": "rf.pkl",
    "KNN": "knn.pkl"
}

models = {}
for name, file in model_files.items():
    with open(MODELS_DIR / file, "rb") as f:
        models[name] = pickle.load(f)


### Features info ###

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


### Sidebar inputs ###

st.sidebar.title("Seoul Bike Demand")

selected_model = st.sidebar.selectbox("Select Model", list(models.keys()), index=3)  # RF default

apply_button = st.sidebar.button("Predict")

user_inputs = {}
# Categorical / integer features
for feature in ["Hour", "Month", "Holiday", "Functioning Day"]:
    min_val, max_val, default = features_info[feature]
    user_inputs[feature] = st.sidebar.selectbox(
        feature,
        options=list(range(int(min_val), int(max_val)+1)),
        index=int(default - min_val)
    )

# Numeric features
for feature in numeric_features:
    min_val, max_val, default = features_info[feature]
    user_inputs[feature] = st.sidebar.slider(feature, float(min_val), float(max_val), float(default))


### Main frame ###

if apply_button:
    input_df = pd.DataFrame([user_inputs])
    input_df = input_df.reindex(columns=feature_order)

    #Prediction
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
        rf_model = models["Random Forest"]
        rf_pred = rf_model.predict(input_df)[0]

        st.subheader("Comparison with Random Forest")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.barplot(
            x=[selected_model, "Random Forest"], 
            y=[pred, rf_pred], 
            palette=["#C60C30", "#003478"],
            ax=ax
        )
        ax.set_ylabel("Predicted Bike Count")
        ax.set_title("Prediction Comparison", fontsize=14)
        st.pyplot(fig)
    
        # Hourly Forecast
    st.subheader("Hourly Forecast for the Day")

    # Create a copy of input_df repeated 24 times
    hourly_df = pd.concat([input_df]*24, ignore_index=True)
    hourly_df["Hour"] = range(24)  # set each row's Hour 0-23
    hourly_df = hourly_df.reindex(columns=feature_order)

    # Predict
    hourly_preds = model.predict(hourly_df)

    # Plot hourly forecast
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(x=range(24), y=hourly_preds, marker="o", color="#C60C30", ax=ax)
    ax.set_xticks(range(0,24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Bike Count")
    ax.set_title(f"Predicted Bike Count Across 24 Hours ({selected_model})")
    st.pyplot(fig)
    


    # Feature importance (RF Only)
    if selected_model == "Random Forest":
      st.subheader("Feature Importance")
      rf_model = models["Random Forest"]
      if hasattr(rf_model, "named_steps"):
          # pipeline case
          rf_model_raw = None
          for step in rf_model.named_steps.values():
              if hasattr(step, "feature_importances_"):
                  rf_model_raw = step
                  break
          if rf_model_raw is None:
              st.error("Random Forest regressor not found in pipeline.")
      else:
          rf_model_raw = rf_model

      if hasattr(rf_model_raw, "feature_importances_"):
          fi = rf_model_raw.feature_importances_
          fi_df = pd.DataFrame({"Feature": feature_order, "Importance": fi})
          fi_df = fi_df.sort_values(by="Importance", ascending=False)

          fig, ax = plt.subplots(figsize=(8,5))
          sns.barplot(
              x="Importance", y="Feature", 
              data=fi_df, 
              palette=sns.color_palette(["#003478", "#FF6B6B", "#C60C30", "#4D79FF"]),
              ax=ax
          )
          ax.set_title("Feature Importance", fontsize=14)
          st.pyplot(fig)
      else:
          st.error("Feature importances not available for Random Forest model.")

    #Supporting Plots
    st.subheader("Supporting Plots")

    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    axes = axes.flatten()

    for i, feature in enumerate(numeric_features):
        temp_df = pd.concat([input_df]*50, ignore_index=True)
        feature_range = pd.Series(
            np.linspace(features_info[feature][0], features_info[feature][1], 50)
        )
        temp_df[feature] = feature_range
        temp_df = temp_df.reindex(columns=feature_order)

        preds = model.predict(temp_df)

        sns.lineplot(x=feature_range, y=preds, color="#C60C30", ax=axes[i%4])
        axes[i%4].set_xlabel(feature)
        axes[i%4].set_ylabel("Predicted Bike Count")
        axes[i%4].set_title(f"{feature} vs Predicted Bikes")

        # When a 2x2 grid is full
        if i % 4 == 3:
            plt.tight_layout(pad=3)
            fig.subplots_adjust(hspace=0.7, wspace=0.5)
            st.pyplot(fig)

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()

    # Handle leftover plots
    if len(numeric_features) % 4 != 0:
        plt.tight_layout(pad=3)
        fig.subplots_adjust(hspace=0.7, wspace=0.5)
        st.pyplot(fig)

    #Manually inputted from notebooks
    model_metrics = {
        "Random Forest": {"R2": 0.8697, "MSE": 54306.23},
        "KNN": {"R2": 0.8095, "MSE": 79375.50},
        "Linear Regression": {"R2": 0.5132, "MSE": 202819.44},
        "Ridge": {"R2": 0.5132, "MSE": 202811.12},
        "Lasso": {"R2": 0.5132, "MSE": 202815.27},
        
    }

    st.divider()
    st.subheader("Model Comparison (Test Set Performance)")

    metrics_df = (
        pd.DataFrame(model_metrics)
        .T
        .reset_index()
        .rename(columns={"index": "Model"})
    )


    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=metrics_df,
        x="Model",
        y="R2",
        color="#003478",
        ax=ax
    )

    ax.set_title("Model R² Comparison")
    ax.set_ylabel("R² Score")
    ax.set_xlabel("")
    plt.xticks(rotation=30)

    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=metrics_df,
        x="Model",
        y="MSE",
        color="#C60C30",
        ax=ax
    )

    ax.set_title("Model Mean Squared Error (Lower is Better)")
    ax.set_ylabel("MSE")
    ax.set_xlabel("")
    plt.xticks(rotation=30)

    st.pyplot(fig)

    st.caption(
        "R² measures how well a model explains overall demand trends. "
        "MSE measures how large prediction errors are on average. "
        "Metrics are computed on a held-out test set."
    )
