import pandas as pd

df = pd.read_csv(
    "data/training_data_2to19.csv"
)

print(df.isnull().sum())

print(df.head())

print(df.shape)

print(df.info())

print(
    "Duplicates:",
    df.duplicated().sum()
)

df = df.drop_duplicates()
df = df.dropna()

print("\nAfter Cleaning:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print(
    df["nutritional_status"]
    .value_counts()
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Features
X = df[
[
    "age_months",
    "gender",
    "height_cm",
    "weight_kg",
    "bmi"
]
]

# Target
y = df["nutritional_status"]

# Convert labels to numbers
le = LabelEncoder()

y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred

)

print("\nConfusion Matrix:")
print(cm)

import pickle

pickle.dump(
    model,
    open("models/malnutrition_model.pkl", "wb")
)

pickle.dump(
    le,
    open("models/label_encoder.pkl", "wb")
)

print("Model Saved Successfully!")
