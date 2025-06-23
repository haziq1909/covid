# train.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

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

# Select features and target
features = ['age', 'comorb', 'vaccinated_dose1', 'vaccinated_dose2',
            'vaccinated_dose3', 'state_code', 'elderly_comorb', 'unvax_comorb',
            'elderly_unvax', 'risk_score']
target = 'bid'

# Impute missing values in features (if any)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(df[features])
y = df[target].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance dataset with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train Logistic Regression
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_res, y_train_res)

# Train XGBoost
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_res, y_train_res)

# Save models and artifacts
joblib.dump(logreg_model, "logreg_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(state_label_map, "state_label_map.pkl")

print("Training complete! Models and artifacts saved.")
