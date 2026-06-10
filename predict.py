import pickle
import pandas as pd

# Load trained model
model = pickle.load(
    open("models/malnutrition_model.pkl", "rb")
)

# Load label encoder
encoder = pickle.load(
    open("models/label_encoder.pkl", "rb")
)

print("=== Malnutrition Prediction System ===")

# User input
age = int(input("Enter Age (months): "))
gender = int(input("Enter Gender (0 = Female, 1 = Male): "))
height = float(input("Enter Height (cm): "))
weight = float(input("Enter Weight (kg): "))
bmi = float(input("Enter BMI: "))

# Create dataframe
sample = pd.DataFrame({
    "age_months": [age],
    "gender": [gender],
    "height_cm": [height],
    "weight_kg": [weight],
    "bmi": [bmi]
})

# Predict
prediction = model.predict(sample)

# Convert numeric prediction to label
result = encoder.inverse_transform(prediction)

print("\n===== RESULT =====")
print("Predicted Nutritional Status:", result[0])