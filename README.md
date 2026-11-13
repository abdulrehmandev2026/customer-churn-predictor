# 📉 Customer Churn Predictor
🔗 **Live Demo:** https://uky9bhvjstrpdtvxaamjrj.streamlit.app/

A machine learning web application that predicts whether a telecom customer will churn (leave the company) from their account and service details. Built entirely in Python using **Streamlit** for the frontend and **scikit-learn** for the ML backend — no separate server required.

---

## 📂 Dataset

**Telco Customer Churn — Kaggle (blastchar/telco-customer-churn)**
- **Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **File used:** `telco_churn.csv` (downloaded from Kaggle with kagglehub)
- **Rows:** 7,043 original → **7,010 after cleaning** (11 blank TotalCharges values removed, 22 duplicate rows removed after dropping customerID)
- **Features:** 19 customer attributes
- **Target:** `Churn` — `0` = Stayed, `1` = Churned
- **Churn rate:** 26.54% before cleaning, **26.49% after cleaning** (1,857 churned / 7,010)

### Feature Descriptions
| Feature | Description |
| --- | --- |
| `gender` | Female / Male |
| `SeniorCitizen` | 1 = senior citizen, 0 = not |
| `Partner`, `Dependents` | Yes / No |
| `tenure` | Months with the company (0–72) |
| `PhoneService`, `MultipleLines` | Phone service details |
| `InternetService` | DSL / Fiber optic / No |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Yes / No / No internet service |
| `Contract` | Month-to-month / One year / Two year |
| `PaperlessBilling` | Yes / No |
| `PaymentMethod` | Bank transfer / Credit card / Electronic check / Mailed check |
| `MonthlyCharges` | Monthly bill in $ (18.25–118.75) |
| `TotalCharges` | Total amount charged in $ (18.8–8,684.8) |

---

## 📋 Project Description

This application provides an end-to-end churn prediction pipeline:

1. **Data Loading** — reads `telco_churn.csv` and shows it with `df.head()`
2. **Cleaning** — drops `customerID`, converts `TotalCharges` to numeric, drops missing values and duplicates
3. **EDA** — churn rate by contract, internet service and payment method; tenure and monthly charges vs churn
4. **Model Training** — trains and evaluates 3 ML classifiers
5. **Model Selection** — picks the best model by ROC-AUC score
6. **Interactive Frontend** — enter customer details → instant churn probability + Low/Moderate/High band

---

## 🛠️ Technologies Used

| Layer | Library | Purpose |
| --- | --- | --- |
| Frontend | `streamlit` | UI, layout, sliders, forms |
| Data | `pandas` | Data loading and manipulation |
| Numerics | `numpy` | Array operations |
| ML Backend | `scikit-learn` | Model training, scaling, evaluation |
| Boosting | `xgboost` | XGBoost classifier |
| Persistence | `joblib` | Saving trained model, scaler and feature lists |

**ML Models trained and compared:**
- Logistic Regression ✅ (Best)
- Random Forest Classifier
- XGBoost Classifier

---

## 📊 Model Performance

Test set: 1,402 customers (80% / 20% stratified split).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | **80.7%** | **66.9%** | **53.4%** | **59.4%** | **0.847** |
| Random Forest | 78.8% | 62.9% | 48.5% | 54.8% | 0.821 |
| XGBoost | 78.5% | 61.3% | 51.2% | 55.8% | 0.823 |

Logistic Regression was picked as the best model because it has the highest ROC-AUC (0.847). Recall is reported because missing a customer who is about to churn is more costly than a false alarm.

### Key findings from EDA
- Month-to-month contracts churn at **42.6%**, vs 11.3% (One year) and 2.8% (Two year)
- Fiber optic customers churn at **41.8%**, vs 18.9% (DSL) and 7.2% (no internet)
- Electronic check payers churn at **45.1%**, the highest of any payment method
- Churned customers stay only **18 months** on average, vs 37.6 months for customers who stay

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- `pip` package manager

### 1. Clone the project
```
git clone https://github.com/abdulrehmandev2026/customer-churn-predictor.git
cd customer-churn-predictor
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

---

## ▶️ Run the Application
```
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure
```
customer-churn-predictor/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependency list
├── telco_churn.csv                 # Dataset
├── churn_model.joblib              # Saved best ML model (Logistic Regression)
├── scaler.joblib                   # Saved StandardScaler
├── feature_columns.joblib          # Saved one-hot encoded feature columns
├── numeric_cols.joblib             # Saved numeric column names
├── Customer_Churn_Predictor.ipynb  # Training notebook
└── README.md                       # This file
```

---

## 🔑 Key Information
| Item | Detail |
| --- | --- |
| Language | Python 3.9+ |
| Framework | Streamlit (frontend + server) |
| Dataset size | 7,010 rows × 20 columns (after cleaning) |
| Train / Test split | 80% / 20% (stratified) |
| Feature encoding | One-hot encoding (30 features) |
| Feature scaling | StandardScaler on tenure, MonthlyCharges, TotalCharges |
| Model selection | Best ROC-AUC on test set |
| Best model | Logistic Regression — 80.7% accuracy, 0.847 ROC-AUC |

---

## ⚕️ Disclaimer
> This tool is intended for **educational and research purposes only**.
> It is **not** a substitute for professional business advice.
> Always verify predictions with real customer data before acting on them.

---
*Built with ❤️ using Python, Scikit-Learn, XGBoost & Streamlit*
