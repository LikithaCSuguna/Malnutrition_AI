import os
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import requests

# Resolve model paths relative to this file for robustness
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "malnutrition_model.pkl")
le_path = os.path.join(base_dir, "models", "label_encoder.pkl")

# Load model safely
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()
    url = "https://your.storage/location/malnutrition_model.pkl"
    r = requests.get(url)
    open(model_path, "wb").write(r.content)
model = pickle.load(open(model_path, "rb"))

# Load encoder if available, else try to construct from model.classes_
encoder = None
if os.path.exists(le_path):
    with open(le_path, "rb") as f:
        encoder = pickle.load(f)
else:
    if hasattr(model, "classes_"):
        encoder = LabelEncoder()
        encoder.classes_ = model.classes_
    else:
        encoder = None

st.set_page_config(
    page_title="Malnutrition Detection",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 AI-Powered Malnutrition Detection System")

st.write("Enter child details below")

age = st.number_input(
    "Age (months)",
    min_value=1,
    max_value=240
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

height = st.number_input(
    "Height (cm)",
    min_value=30.0,
    max_value=250.0
)

weight = st.number_input(
    "Weight (kg)",
    min_value=1.0,
    max_value=200.0
)

if st.button("Predict"):

    gender_value = 0 if gender == "Female" else 1

    height_m = height / 100
    bmi = weight / (height_m ** 2)

    sample = pd.DataFrame({
        "age_months": [age],
        "gender": [gender_value],
        "height_cm": [height],
        "weight_kg": [weight],
        "bmi": [bmi]
    })

    prediction = model.predict(sample)

    result = encoder.inverse_transform(prediction)

    st.success(
        f"Predicted Nutritional Status: {result[0]}"
    )

    st.write(f"Calculated BMI: {bmi:.2f}")