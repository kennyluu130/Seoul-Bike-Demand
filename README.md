# Seoul Bike Demand Prediction

This project predicts the number of rented bikes in Seoul based on environmental, temporal, and holiday-related features. The predictions are made using multiple machine learning models, and a Streamlit app provides an interactive interface for exploring predictions and insights.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Features](#features)
- [Models](#models)
- [Streamlit App](#streamlit-app)
- [Installation](#installation)
- [Usage](#usage)

---

## Project Overview

This project aims to provide accurate bike rental predictions in Seoul to help users make informed decisions about biking. Users can:

- Select a model to predict bike demand.
- Adjust input features like temperature, humidity, hour, and month.
- Visualize predictions and supporting graphs.
- Compare different models in terms of performance metrics.

---

## Data

The dataset includes hourly bike rental counts along with environmental and temporal features. Key data columns:

- `Date`
- `Rented Bike Count`
- `Hour`
- `Temperature(°C)`
- `Humidity(%)`
- `Wind speed (m/s)`
- `Visibility (10m)`
- `Dew point temperature(°C)`
- `Solar Radiation (MJ/m2)`
- `Rainfall(mm)`
- `Snowfall (cm)`
- `Month`
- `Holiday`
- `Functioning Day`

---

## Features

The project uses the following features:

- Numeric: Hour, Temperature, Humidity, Wind speed, Visibility, Dew point, Solar Radiation, Rainfall, Snowfall
- Categorical: Month, Holiday, Functioning Day

All numeric features are scaled for models except Random Forest.

---

## Models

The project includes the following models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- K-Nearest Neighbors

Performance metrics for each model (example):

| Model               | MSE     | RMSE   | MAE    | R²     |
| ------------------- | ------- | ------ | ------ | ------ |
| Linear Regression   | 202,819 | 450.35 | 336.93 | 0.5132 |
| Ridge               | 202,811 | 450.35 | 336.83 | 0.5132 |
| Lasso               | 202,815 | 450.35 | 336.70 | 0.5132 |
| Random Forest       | 54,306  | 233.04 | 139.80 | 0.8697 |
| K-Nearest Neighbors | 79,375  | 281.74 | 173.95 | 0.8095 |

> Random Forest performs best in this dataset.

---

## Streamlit App

The interactive Streamlit app allows users to:

- Select a model (Random Forest default)
- Input or adjust feature values
- View predictions and traffic recommendations
- Compare Random Forest predictions with other models
- Visualize feature importance (for Random Forest)
- See supporting plots and predicted bike demand across 24 hours

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kennyluu130/Seoul-Bike-Demand.git
cd Seoul-Bike-Demand
```

2. Create a virtual environment (Python 3.11)

```bash
python -m venv venv
```

3. Activate the environment

4. Install dependencies

```bash
pip install -r requirements.txt
```

5. Install Git LFS (For Random Forest Model)

```bash
git lfs install
git lfs pull
```

## Usage

Run the Streamlit app:

streamlit run app/main.py
