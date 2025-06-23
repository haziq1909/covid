import streamlit as st
import pickle
import numpy as np

# Load saved model
with open('rf_smote_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title("COVID BID Prediction App")

# Input features
age = st.number_input('Age', min_value=0, max_value=120, value=30)
state = st.selectbox('State', options=['State1', 'State2', 'State3'])  # adjust states
comorb = st.selectbox('Comorbidity (0=No, 1=Yes)', [0,1])
malaysian = st.selectbox('Malaysian (0=No, 1=Yes)', [0,1])
male = st.selectbox('Male (0=Female, 1=Male)', [0,1])
vaccinated_dose1 = st.selectbox('Vaccinated Dose 1 (0=No, 1=Yes)', [0,1])
vaccinated_dose2 = st.selectbox('Vaccinated Dose 2 (0=No, 1=Yes)', [0,1])
vaccinated_dose3 = st.selectbox('Vaccinated Dose 3 (0=No, 1=Yes)', [0,1])

# Map state to numeric value (update this mapping based on your data)
state_mapping = {'State1': 0, 'State2': 1, 'State3': 2}
state_num = state_mapping[state]

# Prepare input for prediction
input_data = np.array([age, state_num, comorb, malaysian, male,
                       vaccinated_dose1, vaccinated_dose2, vaccinated_dose3]).reshape(1, -1)

# Predict button
if st.button('Predict BID Risk'):
    prob = model.predict_proba(input_data)[0][1]  # probability of class 1
    threshold = 0.3  # threshold chosen based on your tuning
    if prob >= threshold:
        st.error(f'High Risk of BID detected! Probability: {prob:.2f}')
    else:
        st.success(f'Low Risk of BID. Probability: {prob:.2f}')

