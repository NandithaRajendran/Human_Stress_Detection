import streamlit as st
import pandas as pd
from backend import hybrid_predict

# App title
st.title("Human Stress Detection using Sleeping Habits 💤")

st.sidebar.header("Enter Your Parameters")

# User input collection
def user_input_features():
    snoring_rate = st.sidebar.text_input('Snoring Rate (0.0 - 10.0)', '2.5')
    respiration_rate = st.sidebar.text_input('Respiration Rate (5.0 - 30.0)', '16.5')
    blood_oxygen_level = st.sidebar.text_input('Blood Oxygen Level (70.0 - 100.0)', '95.0')
    sleep_duration = st.sidebar.text_input('Sleep Duration (0.0 - 12.0 hours)', '7.0')
    heart_rate = st.sidebar.text_input('Heart Rate (40 - 120 bpm)', '70')

    data = {
        'sr': [float(snoring_rate)],
        'rr': [float(respiration_rate)],
        't': [float(blood_oxygen_level)],
        'sl': [float(sleep_duration)],
        'hr': [float(heart_rate)]
    }
    return pd.DataFrame(data)

# Stress level mapping
def map_stress_level(pred):
    if pred == 0:
        return "Normal Stress"
    else:
        return "High Stress"

# Get user input
input_df = user_input_features()

# Show input
st.subheader("Your Input Parameters")
st.write(input_df)

# Predict button
if st.button("Predict"):
    prediction = hybrid_predict(input_df)
    stress_level = map_stress_level(prediction)

    st.subheader("Stress Level Prediction")
    st.write(f"Your predicted stress level is: **{stress_level}**")

    # Optional feedback
    if stress_level == "High Stress":
        st.warning("⚠️ High stress detected! Consider consulting a healthcare professional.")
    else:
        st.success("Low stress! Keep up the good work with your sleep habits.")