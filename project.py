# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import streamlit as st

# Load dataset and preprocessing only once at start
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

df = load_data()

# Buat peta negeri ke integer untuk training model
state_label_map = {state: i for i, state in enumerate(df['state'].unique())}
df['state_code'] = df['state'].map(state_label_map)

features = ['age', 'comorb', 'vaccinated_dose1', 'vaccinated_dose2',
            'vaccinated_dose3', 'state_code', 'elderly_comorb', 'unvax_comorb', 'elderly_unvax', 'risk_score']
target = 'bid'

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(df[features])
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Logistic Regression Model
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_res, y_train_res)
y_pred_logreg = logreg_model.predict(X_test)
y_prob_logreg = logreg_model.predict_proba(X_test)[:, 1]

# XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# --- Sidebar for navigation ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Uji Model"])

if page == "Home":
    st.title("🚑 Analisis Kematian COVID-19 (BID)")
    st.write("**Sila scroll untuk lihat visualisasi dan laporan penuh model.**")

    st.subheader("📂 Dataset Asal")
    st.dataframe(df.head())

    st.subheader("📊 Taburan Umur mengikut Status BID")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x='age', hue='bid', bins=30, kde=True, ax=ax1)
    ax1.set_title('Age Distribution by BID Status')
    st.pyplot(fig1)

    st.subheader("🔢 Peratus BID")
    bid_counts = df['bid'].value_counts(normalize=True)
    st.write(bid_counts)

    st.subheader("📌 Komorbiditi vs BID")
    comorb_bid = pd.crosstab(df['comorb'], df['bid'], normalize='index')
    st.write(comorb_bid)

    st.subheader("🗺️ Negeri dengan Kadar BID Tertinggi")
    state_bid_rate = df.groupby('state')['bid'].mean().sort_values(ascending=False)
    st.write(state_bid_rate.head(10))

    st.subheader("🚀 Logistic Regression - Classification Report")
    st.text(classification_report(y_test, y_pred_logreg))

    cm_logreg = confusion_matrix(y_test, y_pred_logreg)
    st.write("Confusion Matrix (Logistic Regression):")
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title("Logistic Regression Confusion Matrix")
    st.pyplot(fig2)

    st.write(f"ROC AUC (Logistic Regression): {roc_auc_score(y_test, y_prob_logreg):.4f}")

    st.subheader("🚀 XGBoost + SMOTE - Classification Report")
    st.text(classification_report(y_test, y_pred_xgb))

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    st.write("Confusion Matrix (XGBoost):")
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax3)
    ax3.set_title("XGBoost + SMOTE Confusion Matrix")
    st.pyplot(fig3)

    st.write(f"ROC AUC (XGBoost): {roc_auc_score(y_test, y_prob_xgb):.4f}")

    st.subheader("📌 Kepentingan Ciri (XGBoost)")
    importances = xgb_model.feature_importances_
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importances, y=features, ax=ax4)
    ax4.set_title("Feature Importance oleh XGBoost")
    st.pyplot(fig4)

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    st.write("📊 Jadual Kepentingan Ciri:")
    st.dataframe(importance_df)

elif page == "Uji Model":
    st.title("🧪 Uji Model Anda Sendiri")

    age_input = st.number_input("Umur", min_value=0, max_value=120, value=60)
    comorb_input_simple = st.selectbox("Komorbiditi", ["Tidak", "Ya"])
    comorb_encoded = 1 if comorb_input_simple == "Ya" else 0

    state_input = st.selectbox("Negeri", options=list(state_label_map.keys()))
    state_encoded = state_label_map[state_input]

    dose1 = st.checkbox("Telah ambil Dos 1")
    dose2 = st.checkbox("Telah ambil Dos 2")
    dose3 = st.checkbox("Telah ambil Dos Penggalak (Booster)")

    elderly_comorb_input = int((age_input >= 60) and (comorb_encoded == 1))
    unvax_comorb_input = int((comorb_encoded == 1) and (not dose1))
    elderly_unvax_input = int((age_input >= 60) and (not dose1))
    risk_score_input = (
        int(age_input >= 60) * 2 +
        comorb_encoded * 2 +
        int(not dose1) +
        int(not dose2) +
        int(not dose3)
    )

    input_data = np.array([[age_input, comorb_encoded,
                            int(dose1), int(dose2), int(dose3), state_encoded,
                            elderly_comorb_input, unvax_comorb_input, elderly_unvax_input, risk_score_input]])

    st.write("Input data yang dihantar ke model:")
    st.dataframe(pd.DataFrame(input_data, columns=features))

    if st.button("🔍 Ramalkan Risiko BID"):
        prediction_prob_xgb = xgb_model.predict_proba(input_data)[0][1]
        prediction_prob_logreg = logreg_model.predict_proba(input_data)[0][1]

        if age_input < 30 and comorb_encoded == 0 and dose1 and dose2:
            prediction_prob_xgb = min(prediction_prob_xgb, 0.2)
            prediction_prob_logreg = min(prediction_prob_logreg, 0.2)

        st.write("🔎 Hasil Ramalan:")
        st.write(f"Probabiliti risiko oleh model XGBoost: {prediction_prob_xgb:.4f}")
        st.write(f"Probabiliti risiko oleh model Logistic Regression: {prediction_prob_logreg:.4f}")
        st.write(f"Risk score input: {risk_score_input}")

        if prediction_prob_xgb >= 0.5 or risk_score_input >= 4:
            st.error(f"⚠️ Individu ini BERISIKO tinggi BID (XGBoost Probabiliti: {prediction_prob_xgb:.2f})")
        else:
            st.success(f"✅ Individu ini TIDAK berisiko tinggi BID (XGBoost Probabiliti: {prediction_prob_xgb:.2f})")
