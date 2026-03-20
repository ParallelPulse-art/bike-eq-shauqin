# 🏍️ Bike Price Prediction App

A full-featured ML-powered Streamlit web app to predict motorcycle resale prices.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place both files in the same folder
```
bike_price_app.py
bike_price_data.csv
requirements.txt
```

### 3. Run
```bash
streamlit run bike_price_app.py
```
Opens at **http://localhost:8501**

---

## 🔢 Prediction Parameters (7)

| # | Feature | Type | Values |
|---|---------|------|--------|
| 1 | **Brand** | Categorical | Honda, Yamaha, KTM, BMW, Ducati, Harley… |
| 2 | **Fuel Type** | Categorical | Petrol / Electric |
| 3 | **Bike Type** | Categorical | Sport, Cruiser, Adventure, Commuter… |
| 4 | **Condition** | Categorical | New / Used / Like New |
| 5 | **Transmission** | Categorical | Manual / Automatic / Semi-Automatic |
| 6 | **Owner Type** | Categorical | 1st / 2nd / 3rd Owner |
| 7 | **Year** | Numeric | 2005 – 2024 |
| + | **Engine CC** | Numeric | 100 – 1340 cc |
| + | **Power (BHP)** | Numeric | 5 – 200 bhp |
| + | **Mileage (kmpl)** | Numeric | 15 – 90 kmpl |
| + | **Kms Driven** | Numeric | 0 – 120,000 km |

---

## 🤖 ML Models

| Model | Notes |
|-------|-------|
| **Gradient Boosting** | Best accuracy — default recommendation |
| **Random Forest** | Robust, handles outliers well |
| **Extra Trees** | Fast, low-bias ensemble |
| **Ridge Regression** | Linear baseline |

---

## 📊 App Tabs (6)

| Tab | Contents |
|-----|----------|
| 📊 Overview | Stats, box plots, pie/donut charts, data table |
| 🔍 EDA | Scatter plots, trend lines, heatmaps, histograms, Seaborn heatmap |
| 🤖 Model Performance | R², MAE, RMSE, MAPE, actual vs predicted, residuals |
| 📈 Feature Importance | Importance bars, insights, partial dependence plots |
| 🔗 Correlations | Correlation matrix, pairplot, bar comparisons |
| 🏷️ Brand Analysis | Per-brand deep dive — models, trends, violin plots |

---

## ☁️ Deploy Free on Streamlit Cloud

1. Push all files to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Select repo, set main file: `bike_price_app.py`
4. Click **Deploy** — live in ~2 minutes!

---

## 📦 Dataset

Generated from real Kaggle bike listing patterns covering:
- 3,000 bike records
- 10 brands · 50 models · 4 fuel types · 6 bike categories
- Price range: ₹8,000 – ₹12,00,000
