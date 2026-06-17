# ⚡ Household Power Consumption Forecasting

> **Time Series Forecasting & Classification** — Comparing LSTM, GRU, Bi-LSTM, and Decision Tree with Shapelet Transform on real-world household electricity data.

---

## 📌 Project Overview

This project analyzes and forecasts daily household power consumption using the [UCI Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) dataset. Multiple deep learning and machine learning models are implemented and compared to evaluate their effectiveness on sequential time series data.

**Key question:** _Which model best captures temporal patterns in electricity usage — and can we classify consumption levels with traditional ML?_

---

## 🗂️ Repository Structure

```
├── power_daily_clean.csv   # Cleaned & resampled daily dataset (preprocessed from raw UCI data)
├── LSTM.ipynb              # LSTM model — baseline deep learning approach
├── GRU.ipynb               # GRU model — lighter alternative to LSTM
├── BLSTM.ipynb             # Bidirectional LSTM — captures past & future context
├── tree.ipynb              # Decision Tree + Shapelet Transform — ML classification baseline
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🔧 Methods & Pipeline

### Data Preprocessing (`LSTM.ipynb` — cleaning section)

- Loaded raw UCI dataset (minute-level readings, ~2M rows)
- Parsed datetime index, handled missing values with forward-fill
- Resampled to **daily mean** `Global_active_power` → `power_daily_clean.csv`

### Forecasting Models

| Model       | Architecture                                 | Epochs | Notes                      |
| ----------- | -------------------------------------------- | ------ | -------------------------- |
| **LSTM**    | 2-layer LSTM (64→32 units) + Dropout         | 10     | Baseline RNN approach      |
| **GRU**     | 2-layer GRU (50→50 units)                    | 10     | Fewer parameters than LSTM |
| **Bi-LSTM** | 2-layer Bidirectional LSTM (50→50) + Dropout | 25     | Best context capture       |

- **Sliding window:** 30-day look-back
- **Train/Test split:** 80/20
- **Scaler:** MinMaxScaler (0–1)
- **Loss:** Mean Squared Error | **Optimizer:** Adam
- **Metrics:** RMSE, MAE, R², MAPE

### Classification Model (`tree.ipynb`)

- Engineered binary label: High/Low consumption (threshold = mean of 7-day windows)
- Applied **Shapelet Transform** to extract discriminative subsequences
- Classified with **Decision Tree** (max_depth=3, entropy criterion)
- Evaluated with accuracy score, confusion matrix, classification report

---

## 📊 Results

> _(Update this section after running all notebooks — paste final RMSE/MAE/R² from each model here for easy comparison)_

| Model         | RMSE         | MAE | R²  |
| ------------- | ------------ | --- | --- |
| LSTM          | —            | —   | —   |
| GRU           | —            | —   | —   |
| Bi-LSTM       | —            | —   | —   |
| Decision Tree | Accuracy: —% | —   | —   |

---

## 🛠️ Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, metrics, Decision Tree
- `pyts` — Shapelet Transform (time series ML)
- `TensorFlow / Keras` — LSTM, GRU, Bidirectional LSTM
- `matplotlib`, `seaborn` — visualization

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/nguyentrong243/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order:
#    LSTM.ipynb → GRU.ipynb → BLSTM.ipynb → tree.ipynb
#    (power_daily_clean.csv is already included — no raw data download needed)
```

---

## 📚 Dataset

- **Source:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power-consumption)
- **Original:** ~2M minute-level readings from a single household in France (2006–2010)
- **Used:** Daily-aggregated `Global_active_power` after preprocessing

---

## 👤 Author

**Nguyen Trong**
