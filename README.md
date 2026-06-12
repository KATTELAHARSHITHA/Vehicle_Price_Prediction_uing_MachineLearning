# Vehicle_Price_Prediction_uing_MachineLearning
# 🚗 Vehicle Price Prediction Using Machine Learning

A machine learning project to predict the resale price of used vehicles using real-world data. The project covers the complete ML pipeline — data loading, EDA, preprocessing, feature engineering, model building, and evaluation.

---

## 📌 Project Overview

| Item | Details |
|---|---|
| Problem Type | Regression |
| Target Variable | Resale_Price |
| Best Model | XGBoost Regressor |
| Best R² Score | 0.94 (test data) |
| Validation | 5-Fold Cross Validation |
| Deployment | Streamlit Web App |

---

## 📂 Dataset

**File:** `real_time_resale_dataset_88.csv`

### Columns

| Column | Type | Description |
|---|---|---|
| Car_Age | Numerical | Age of the car in years |
| Kms_Driven | Numerical | Total kilometers driven |
| Mileage_kmpl | Numerical | Fuel efficiency in km per litre |
| Displacement_cc | Numerical | Engine size in cubic centimeters |
| Gears | Numerical | Number of gears |
| Safety_Rating | Numerical | Overall safety score |
| Fuel_Type | Categorical | Petrol / Diesel / CNG / Electric |
| ABS | Binary (Yes/No) | Anti-lock Braking System |
| Traction_Control | Binary (Yes/No) | Traction Control feature |
| Cruise_Control | Binary (Yes/No) | Cruise Control feature |
| Power_Steering | Binary (Yes/No) | Power Steering feature |
| Resale_Price | Target | Resale price of the vehicle (₹) |

---

## 🔍 Exploratory Data Analysis

### Checks Performed
- `data.info()` — data types and null counts
- `data.describe()` — statistical summary
- `data.isnull().sum()` — missing value check
- `data.duplicated().sum()` — duplicate record check

### Visualisations
- **Histograms** — distribution of all numerical features
- **Scatter plot** — Car_Age vs Resale_Price
- **Box plot** — Fuel_Type vs Resale_Price
- **Correlation Heatmap** — relationships between numerical features
- **Box plots** — outlier detection for all numerical columns

---

## 🧹 Data Preprocessing

### 1. Data Type Conversion
```python
data['Mileage_kmpl'] = data['Mileage_kmpl'].astype(int)
```

### 2. Log Transformation
Applied to reduce skewness in right-skewed features:
```python
data['Resale_Price_log'] = np.log1p(data['Resale_Price'])
data['Mileage_log']      = np.log1p(data['Mileage_kmpl'])
```

### 3. Outlier Capping — IQR Method
Outliers capped for these columns:
```python
outlier_cols = ['Resale_Price', 'Mileage_kmpl', 'Displacement_cc', 'Kms_Driven']
```
Formula:
```
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
Values outside bounds are capped — not removed
```

### 4. Binary Encoding (Yes/No → 1/0)
```python
binary_cols = ['Cruise_Control', 'Traction_Control', 'ABS', 'Power_Steering']
```

### 5. Numeric Type Fixing
```python
numeric_object_cols = ['Safety_Rating', 'Gears']
X[col] = pd.to_numeric(X[col], errors='coerce')
```

---

## ⚙️ Feature Engineering

### ColumnTransformer
Different preprocessing applied to different column types:

```python
transformer = ColumnTransformer([
    ('num', StandardScaler(),              num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
```

| Column Type | Columns | Transformation |
|---|---|---|
| Numerical | Car_Age, Kms_Driven, Mileage_kmpl, Displacement_cc, Gears, Safety_Rating | StandardScaler |
| Categorical | Fuel_Type | OneHotEncoder |
| Binary | ABS, Traction_Control, Cruise_Control, Power_Steering | Mapped to 1/0 |

---

## 🤖 Model Building

### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Training set: 80%
- Test set: 20%

### Models Trained

#### 1. Linear Regression (Baseline)
```python
pipe = Pipeline([
    ("preprocessing", transformer),
    ("model", LinearRegression())
])
```

#### 2. XGBoost Regressor
```python
pipe2 = Pipeline([
    ("preprocessing", transformer),
    ("model", XGBRegressor())
])
```

#### 3. LightGBM Regressor
```python
pipe3 = Pipeline([
    ("preprocessing", transformer),
    ("model", LGBMRegressor())
])
```

> **Why Pipeline?** Combines preprocessing and model in one step — prevents data leakage by ensuring the scaler is fit only on training data.

---

## 📊 Model Evaluation

### Metric Used — R² Score
> Measures how much variance in resale price the model explains.
> R² = 1.0 is perfect. R² = 0 means the model is no better than predicting the mean.

### Cross Validation
```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe2, X, y, cv=cv, scoring='r2', n_jobs=-1)
```
- 5-fold cross validation used for both XGBoost and LightGBM
- Ensures model performance is consistent across different data splits

### Results

| Model | Test R² Score | Validation |
|---|---|---|
| Linear Regression | Baseline | — |
| XGBoost | **0.94** ✅ | 5-Fold CV |
| LightGBM | Comparable | 5-Fold CV |

> XGBoost achieved **0.94 R²** on unseen test data — explaining 94% of the variation in vehicle resale prices.

---

## 🗂️ Project Structure

```
vehicle-price-prediction/
│
├── vehicle_price_pred.ipynb     ← Main Jupyter notebook
├── real_time_resale_dataset_88.csv  ← Dataset
├── app.py                       ← Streamlit deployment app
├── vehicle_model.pkl            ← Saved trained model
└── requirements.txt             ← Dependencies
```

---

## 🚀 Deployment

The model is deployed as an interactive web app using **Streamlit**.

Users can input:
- Car Age, Kilometers Driven, Mileage
- Fuel Type, Engine Displacement
- Safety features (ABS, Traction Control, etc.)

And get an instant **predicted resale price**.

### Run Locally
```bash
pip install streamlit
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Deployment | Streamlit |
| Model Saving | Joblib |

---

## 📈 Key Findings

- **Car_Age and Kms_Driven** are negatively correlated with resale price — older and more used cars are cheaper
- **Displacement_cc** is positively correlated — bigger engine cars tend to be premium
- **Fuel_Type** significantly affects price — Diesel preferred for high mileage buyers
- **XGBoost outperformed Linear Regression** — confirms non-linear relationships in the data
- **Log transformation** on Resale_Price and Mileage improved model stability

---


**Kattela Harshitha**
Data Science Graduate — MGIT Hyderabad
[LinkedIn](https://www.linkedin.com/in/kattelaharshitha/)
