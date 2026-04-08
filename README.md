# Seoul Bike Demand Prediction

Seoul operates one of the world's largest public bicycle-sharing schemes. Accurately forecasting hourly demand lets city planners **pre-position bikes**, **optimize maintenance schedules**, and **improve rider experience**.

This project builds and compares **seven regression models**, from simple linear baselines to gradient-boosted trees, trained on two full years of hourly sensor data covering weather, time-of-day, and calendar features.

---

## 📌 Project Overview

This project provides an interactive platform to explore bike rental patterns in Seoul and generate real-time predictions.

* **Model Selection:** Choose from 7 different algorithms to see how they handle demand.
* **Dynamic Inputs:** Adjust temperature, humidity, and time to see immediate prediction changes.
* **Deep Insights:** Visualize commuter peaks, seasonal swings, and feature importance.
* **Performance Benchmarking:** Compare tree-based models against linear baselines.

---

## 📊 Analytic Insights

Our analysis revealed several critical patterns in how Seoul uses its bike-sharing system:

| Finding | Detail |
| :--- | :--- |
| **Top Predictor** | **Temperature** (demand rises sharply above 10°C). |
| **Commuter Double-Peak** | Strong **08:00** and **18:00** spikes on weekdays; weekends peak mid-afternoon. |
| **Seasonal Swing** | Summer demand is ~3x higher than winter. |
| **Rain Effect** | Rentals drop sharply during any rainfall hours. |
| **Best Model** | **XGBoost** ($R^2$ 0.877), followed closely by LightGBM. |
| **Linear Ceiling** | Linear models cap at $R^2 \approx 0.42$, confirming strong non-linearity in the data. |

---

## 🧪 Models & Performance

We evaluated the models using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$).

| Model | MSE | RMSE | MAE | $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **51,221.19** | **226.32** | **126.97** | **0.8771** |
| **LightGBM** | 53,868.87 | 232.10 | 130.76 | 0.8707 |
| **Random Forest** | 61,793.77 | 248.58 | 141.15 | 0.8517 |
| **K-Nearest Neighbors** | 89,352.73 | 298.92 | 171.79 | 0.7855 |
| **Lasso Regression** | 239,240.59 | 489.12 | 308.74 | 0.4258 |
| **Ridge Regression** | 240,103.23 | 490.00 | 309.15 | 0.4237 |
| **Linear Regression** | 240,392.03 | 490.30 | 309.29 | 0.4230 |

### 💡 Key Takeaways
* **Gradient Boosting Wins:** XGBoost and LightGBM achieve the best performance, explaining roughly **87%** of the variance in hourly demand.
* **Tree-Based Superiority:** The most impactful improvement came from switching from linear to tree-based methods, which handle the non-linear relationship between weather and demand much more effectively.
* **Interpretability:** While XGBoost wins on metrics, **Random Forest** remains highly valuable for providing clear feature importance rankings.

---

## 📂 Data & Features

The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand).

### Key Features:
* **Temporal:** Hour, Month, Holiday, Functioning Day.
* **Weather:** Temperature (°C), Humidity (%), Wind speed (m/s), Visibility (10m), Dew point temperature (°C), Solar Radiation (MJ/m2), Rainfall (mm), Snowfall (cm).

---

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kennyluu130/Seoul-Bike-Demand.git](https://github.com/kennyluu130/Seoul-Bike-Demand.git)
    cd Seoul-Bike-Demand
    ```

2.  **Set up the environment (Python 3.11):**
    ```bash
    python -m venv venv
    # Activate venv:
    # Windows: .\venv\Scripts\activate | Mac/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Git LFS (Required for model files):**
    ```bash
    git lfs install
    git lfs pull
    ```

## 💻 Usage

Launch the interactive dashboard to explore the data and generate predictions:

```bash
streamlit run app/main.py
