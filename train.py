# train.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("covid_deaths_linelist.csv")
df.columns = df.columns.str.lower()
df['bid'] = df['bid'].astype(int)

# Handle missing values & preprocess
df['comorb'] = df['comorb'].fillna(0).astype(int)
placeholders = ["Pending VMS sync", "Unknown", "Not Available", "N/A", "NA"]
for placeholder in placeholders:
    df.replace(placeholder, np.nan, inplace=True)

df['age'] = df['age'].fillna(df['age'].median())
for col in ['sex', 'state', 'citizenship', 'comorb']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Feature engineering
df['vaccinated_dose1'] = df['date_dose1'].notnull().astype(int)
df['vaccinated_dose2'] = df['date_dose2'].notnull().astype(int)
df['vaccinated_dose3'] = df['date_dose3'].notnull().astype(int)
df['elderly_comorb'] = ((df['age'] >= 60) & (df['comorb'] == 1)).astype(int)
df['unvax_comorb'] = ((df['comorb'] == 1) & (df['vaccinated_dose1'] == 0)).astype(int)
df['elderly_unvax'] = ((df['age'] >= 60) & (df['vaccinated_dose1'] == 0)).astype(int)
df['risk_score'] = (
    (df['age'] >= 60).astype(int) * 2 +
    df['comorb'] * 2 +
    (df['vaccinated_dose1'] == 0).astype(int) +
    (df['vaccinated_dose2'] == 0).astype(int) +
    (df['vaccinated_dose3'] == 0).astype(int)
)

# Map states to integers
state_label_map = {state: i for i, state in enumerate(df['state'].unique())}
df['state_code'] = df['state'].map(state_label_map)

# Full feature set
all_features = [
    'age', 'comorb', 'vaccinated_dose1', 'vaccinated_dose2',
    'vaccinated_dose3', 'state_code', 'elderly_comorb',
    'unvax_comorb', 'elderly_unvax', 'risk_score'
]
target = 'bid'

# Impute full features
imputer_full = SimpleImputer(strategy='median')
X_all = imputer_full.fit_transform(df[all_features])
y = df[target].values

# Split full dataset
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Balance dataset with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train full logistic regression model
logreg_full = LogisticRegression(max_iter=1000, random_state=42)
logreg_full.fit(X_train_res, y_train_res)

# Save full imputer and full logistic regression model
joblib.dump(imputer_full, "models/imputer_full.pkl")
joblib.dump(logreg_full, "models/logreg_full.pkl")

# Train XGBoost model full features
xgb_full = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_full.fit(X_train_res, y_train_res)

# Save XGBoost full model
joblib.dump(xgb_full, "models/xgb_full.pkl")


# Feature selection based on coefficients >= 0.4 (absolute value)
coefs = logreg_full.coef_[0]
abs_coefs = np.abs(coefs)
selected_features = [feat for feat, coef in zip(all_features, abs_coefs) if coef >= 0.4]

print("✅ Selected features based on Logistic Regression coef ≥ 0.4:")
print(selected_features)

# Save selected features list
joblib.dump(selected_features, "models/selected_features.pkl")

# Imputer for selected features only
imputer_selected = SimpleImputer(strategy='median')
X_selected = imputer_selected.fit_transform(df[selected_features])

# Split dataset with selected features
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# SMOTE for selected features
X_train_sel_res, y_train_sel_res = smote.fit_resample(X_train_sel, y_train_sel)

# Train logistic regression model with selected features
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_sel_res, y_train_sel_res)

# Train XGBoost model with selected features
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_sel_res, y_train_sel_res)

# Save models and artifacts for selected features
joblib.dump(logreg_model, "models/logreg_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(imputer_selected, "models/imputer.pkl")  # overwrite imputer.pkl for selected features
joblib.dump(state_label_map, "models/state_label_map.pkl")

print("✅ Training complete with selected features. All models and files saved in /models/")
