import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

@st.cache_data
def load_data():
    df = pd.read_csv("covid_deaths_linelist.csv")
    df.columns = df.columns.str.lower()
    df['bid'] = df['bid'].astype(int)
    df['comorb'] = df['comorb'].fillna(0).astype(int)
    df['brand1'] = df['brand1'].fillna('Unknown')

    placeholders = ["Pending VMS sync", "Unknown", "Not Available", "N/A", "NA"]
    for placeholder in placeholders:
        df.replace(placeholder, np.nan, inplace=True)

    if 'age' in df.columns:
        df['age'] = df['age'].fillna(df['age'].median())

    cat_cols = ['sex', 'state', 'citizenship', 'comorb']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

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
    return df

# Load data
df = load_data()

# Mapping states to integers
state_label_map = {state: i for i, state in enumerate(df['state'].unique())}
df['state_code'] = df['state'].map(state_label_map)

# Full and selected feature sets
full_features = [
    'age', 'comorb', 'vaccinated_dose1', 'vaccinated_dose2',
    'vaccinated_dose3', 'state_code', 'elderly_comorb',
    'unvax_comorb', 'elderly_unvax', 'risk_score'
]

selected_features = joblib.load("models/selected_features.pkl")

target = 'bid'
y = df[target].values

# Load imputers
imputer_full = joblib.load("models/imputer_full.pkl")
imputer_selected = joblib.load("models/imputer.pkl")

# Load models
logreg_full = joblib.load("models/logreg_full.pkl")
xgb_full = joblib.load("models/xgb_full.pkl")  # Pastikan ada model ini
logreg_selected = joblib.load("models/logreg_model.pkl")
xgb_selected = joblib.load("models/xgb_model.pkl")

st.title("🚑 COVID-19 BID Analysis - Model Comparison")

st.subheader("📂 Dataset Preview (5 Rows)")
st.dataframe(df.head())

# Prepare feature matrices
X_full = imputer_full.transform(df[full_features])
X_selected = imputer_selected.transform(df[selected_features])

# Predictions and probabilities - full model (before retrain)
y_pred_logreg_full = logreg_full.predict(X_full)
y_prob_logreg_full = logreg_full.predict_proba(X_full)[:, 1]
y_pred_xgb_full = xgb_full.predict(X_full)
y_prob_xgb_full = xgb_full.predict_proba(X_full)[:, 1]

# Predictions and probabilities - selected model (after retrain)
y_pred_logreg_sel = logreg_selected.predict(X_selected)
y_prob_logreg_sel = logreg_selected.predict_proba(X_selected)[:, 1]

y_pred_xgb_sel = xgb_selected.predict(X_selected)
y_prob_xgb_sel = xgb_selected.predict_proba(X_selected)[:, 1]

# Display classification reports
st.subheader("📊 Logistic Regression - Before Feature Selection (Full Features)")
st.text(classification_report(y, y_pred_logreg_full))

st.subheader("📊 XGBoost - Before Feature Selection (Full Features)")
st.text(classification_report(y, y_pred_xgb_full))

st.subheader("📊 Logistic Regression - After Feature Selection (Selected Features)")
st.text(classification_report(y, y_pred_logreg_sel))

st.subheader("📊 XGBoost - After Feature Selection (Selected Features)")
st.text(classification_report(y, y_pred_xgb_sel))

# Display ROC AUC scores
st.write(f"ROC AUC Logistic Regression (Full features): {roc_auc_score(y, y_prob_logreg_full):.4f}")
st.write(f"ROC AUC XGBoost (Full features): {roc_auc_score(y, y_prob_xgb_full):.4f}")
st.write(f"ROC AUC Logistic Regression (Selected features): {roc_auc_score(y, y_prob_logreg_sel):.4f}")
st.write(f"ROC AUC XGBoost (Selected features): {roc_auc_score(y, y_prob_xgb_sel):.4f}")

# Confusion matrices side by side
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

cm_logreg_full = confusion_matrix(y, y_pred_logreg_full)
sns.heatmap(cm_logreg_full, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Confusion Matrix\nLogReg Full Features")

cm_xgb_full = confusion_matrix(y, y_pred_xgb_full)
sns.heatmap(cm_xgb_full, annot=True, fmt='d', cmap='Purples', ax=axes[1])
axes[1].set_title("Confusion Matrix\nXGBoost Full Features")

cm_logreg_sel = confusion_matrix(y, y_pred_logreg_sel)
sns.heatmap(cm_logreg_sel, annot=True, fmt='d', cmap='Greens', ax=axes[2])
axes[2].set_title("Confusion Matrix\nLogReg Selected Features")

cm_xgb_sel = confusion_matrix(y, y_pred_xgb_sel)
sns.heatmap(cm_xgb_sel, annot=True, fmt='d', cmap='Oranges', ax=axes[3])
axes[3].set_title("Confusion Matrix\nXGBoost Selected Features")

st.pyplot(fig)

# Feature importance - Logistic Regression (Full Features)
coef_full = logreg_full.coef_[0]
importance_logreg_full = pd.DataFrame({
    'Feature': full_features,
    'Coefficient': coef_full,
    'Abs_Coefficient': np.abs(coef_full)
}).sort_values(by='Abs_Coefficient', ascending=False)

st.subheader("📌 Feature Importance - Logistic Regression (Full Features)")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x='Abs_Coefficient', y='Feature', data=importance_logreg_full, ax=ax2)
ax2.set_title("Logistic Regression Coefficients (Full Features, |coef|)")
st.pyplot(fig2)
st.dataframe(importance_logreg_full)

# Feature importance - XGBoost (Full Features)
importance_xgb_full = pd.DataFrame({
    'Feature': full_features,
    'Importance': xgb_full.feature_importances_
}).sort_values(by='Importance', ascending=False)

st.subheader("📌 Feature Importance - XGBoost (Full Features)")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_xgb_full, ax=ax3)
ax3.set_title("XGBoost Feature Importance (Full Features)")
st.pyplot(fig3)
st.dataframe(importance_xgb_full)

# # Feature importance - Logistic Regression (Selected Features)
# coef_sel = logreg_selected.coef_[0]
# importance_logreg_sel = pd.DataFrame({
#     'Feature': selected_features,
#     'Coefficient': coef_sel,
#     'Abs_Coefficient': np.abs(coef_sel)
# }).sort_values(by='Abs_Coefficient', ascending=False)

# st.subheader("📌 Feature Importance - Logistic Regression (Selected Features)")
# fig4, ax4 = plt.subplots(figsize=(8, 6))
# sns.barplot(x='Abs_Coefficient', y='Feature', data=importance_logreg_sel, ax=ax4)
# ax4.set_title("Logistic Regression Coefficients (Selected Features, |coef|)")
# st.pyplot(fig4)
# st.dataframe(importance_logreg_sel)

# # Feature importance - XGBoost (Selected Features)
# importance_xgb_sel = pd.DataFrame({
#     'Feature': selected_features,
#     'Importance': xgb_selected.feature_importances_
# }).sort_values(by='Importance', ascending=False)

# st.subheader("📌 Feature Importance - XGBoost (Selected Features)")
# fig5, ax5 = plt.subplots(figsize=(8, 6))
# sns.barplot(x='Importance', y='Feature', data=importance_xgb_sel, ax=ax5)
# ax5.set_title("XGBoost Feature Importance (Selected Features)")
# st.pyplot(fig5)
# st.dataframe(importance_xgb_sel)
