import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load models and selected features list
logreg_model = joblib.load("models/logreg_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
imputer = joblib.load("models/imputer.pkl")
state_label_map = joblib.load("models/state_label_map.pkl")
selected_features = joblib.load("models/selected_features.pkl")  # Load selected features list

st.title("🧪 Uji Model Risiko Kematian COVID-19 (BID)")

# Input user
age_input = st.number_input("Umur", min_value=0, max_value=120, value=60)
comorb_input = st.selectbox("Adakah anda mempunyai komorbiditi?", ["Tidak", "Ya"])
comorb_encoded = 1 if comorb_input == "Ya" else 0

state_input = st.selectbox("Negeri", options=list(state_label_map.keys()))
state_encoded = state_label_map[state_input]

dose1 = st.checkbox("Telah ambil Dos 1 vaksin")
dose2 = st.checkbox("Telah ambil Dos 2 vaksin")
dose3 = st.checkbox("Telah ambil Dos Penggalak (Booster)")

# Feature engineering input sama seperti train.py
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

# Buat dict dengan semua features penuh
full_input_dict = {
    'age': age_input,
    'comorb': comorb_encoded,
    'vaccinated_dose1': int(dose1),
    'vaccinated_dose2': int(dose2),
    'vaccinated_dose3': int(dose3),
    'state_code': state_encoded,
    'elderly_comorb': elderly_comorb_input,
    'unvax_comorb': unvax_comorb_input,
    'elderly_unvax': elderly_unvax_input,
    'risk_score': risk_score_input
}

# Buat DataFrame input
input_df = pd.DataFrame([full_input_dict])

# Ambil subset hanya selected features untuk imputer & model
input_selected = input_df[selected_features]

st.write("Data input kepada model (selected features):")
st.dataframe(input_selected)

if st.button("🔍 Ramalkan Risiko BID"):
    # Impute input data
    input_imputed = imputer.transform(input_selected)

    # Predict probabiliti dengan model XGBoost dan Logistic Regression
    pred_prob_xgb = xgb_model.predict_proba(input_imputed)[0][1]
    pred_prob_logreg = logreg_model.predict_proba(input_imputed)[0][1]

    # Adjustment khusus (boleh ubah ikut keperluan)
    if age_input < 30 and comorb_encoded == 0 and dose1 and dose2:
        pred_prob_xgb = min(pred_prob_xgb, 0.2)
        pred_prob_logreg = min(pred_prob_logreg, 0.2)

    st.write("### Hasil Ramalan Risiko BID:")
    st.write(f"Probabiliti risiko oleh XGBoost: {pred_prob_xgb:.4f}")
    st.write(f"Probabiliti risiko oleh Logistic Regression: {pred_prob_logreg:.4f}")
    st.write(f"Risk score input: {risk_score_input}")

    if pred_prob_xgb >= 0.5 or risk_score_input >= 4:
        st.error(f"⚠️ Risiko tinggi BID (XGBoost Probabiliti: {pred_prob_xgb:.2f})")
    else:
        st.success(f"✅ Risiko rendah BID (XGBoost Probabiliti: {pred_prob_xgb:.2f})")
