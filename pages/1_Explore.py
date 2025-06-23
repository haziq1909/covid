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

# Load dataset
df = load_data()

# Load models and imputer for classification reports etc.
logreg_model = joblib.load("models/logreg_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
imputer = joblib.load("models/imputer.pkl")

# Mapping negeri ke integer
state_label_map = {state: i for i, state in enumerate(df['state'].unique())}
df['state_code'] = df['state'].map(state_label_map)

features = ['age', 'comorb', 'vaccinated_dose1', 'vaccinated_dose2',
            'vaccinated_dose3', 'state_code', 'elderly_comorb', 'unvax_comorb', 'elderly_unvax', 'risk_score']
target = 'bid'

# Impute features and prepare X, y for evaluation
X = imputer.transform(df[features])
y = df[target].values

# Predict with loaded models
y_pred_logreg = logreg_model.predict(X)
y_prob_logreg = logreg_model.predict_proba(X)[:, 1]
y_pred_xgb = xgb_model.predict(X)
y_prob_xgb = xgb_model.predict_proba(X)[:, 1]

st.title("🚑 Analisis Kematian COVID-19 (BID) - Dashboard Data & Model")

st.subheader("📂 Dataset Asal (Contoh 5 Baris)")
st.dataframe(df.head())

st.subheader("📊 Taburan Umur mengikut Status BID")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x='age', hue='bid', bins=30, kde=True, ax=ax1)
ax1.set_title('Taburan Umur mengikut Status BID')
st.pyplot(fig1)

st.subheader("🔢 Peratus BID")
bid_counts = df['bid'].value_counts(normalize=True)
st.write(bid_counts)

st.subheader("📌 Komorbiditi vs BID")
comorb_bid = pd.crosstab(df['comorb'], df['bid'], normalize='index')
st.write(comorb_bid)

st.subheader("🗺️ Negeri dengan Kadar BID Tertinggi")
state_bid_rate = df.groupby('state')['bid'].mean().sort_values(ascending=False)
st.write(state_bid_rate)

st.subheader("🚀 Logistic Regression - Classification Report")
st.text(classification_report(y, y_pred_logreg))

cm_logreg = confusion_matrix(y, y_pred_logreg)
st.write("Confusion Matrix (Logistic Regression):")
fig2, ax2 = plt.subplots()
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title("Confusion Matrix Logistic Regression")
st.pyplot(fig2)

st.write(f"ROC AUC (Logistic Regression): {roc_auc_score(y, y_prob_logreg):.4f}")

st.subheader("🚀 XGBoost + SMOTE - Classification Report")
st.text(classification_report(y, y_pred_xgb))

cm_xgb = confusion_matrix(y, y_pred_xgb)
st.write("Confusion Matrix (XGBoost):")
fig3, ax3 = plt.subplots()
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax3)
ax3.set_title("Confusion Matrix XGBoost + SMOTE")
st.pyplot(fig3)

st.write(f"ROC AUC (XGBoost): {roc_auc_score(y, y_prob_xgb):.4f}")

st.subheader("📌 Kepentingan Ciri (XGBoost)")
importances = xgb_model.feature_importances_
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.barplot(x=importances, y=features, ax=ax4)
ax4.set_title("Kepentingan Ciri oleh XGBoost")
st.pyplot(fig4)

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
st.write("📊 Jadual Kepentingan Ciri:")
st.dataframe(importance_df)
