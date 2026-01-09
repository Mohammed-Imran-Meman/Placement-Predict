import streamlit as st
import pickle
import numpy as np

# Load model and scaler
clf = pickle.load(open("placement_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Placement Prediction App")

iq = st.number_input("IQ", min_value=0.0, max_value=200.0, step=1.0)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict Placement"):
    input_data = np.array([[iq, cgpa]])
    input_scaled = scaler.transform(input_data)

    prediction = clf.predict(input_scaled)[0]
    prob = clf.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success("Student will get PLACED")
    else:
        st.error("Student will NOT get placed")
    st.info(f"Placement Probability: {prob*100:.2f}%")