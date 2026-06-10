# AI-Powered Malnutrition Detection System

A Machine Learning-based healthcare application that predicts nutritional status using child growth measurements.

## Features

* Malnutrition Detection
* BMI Calculation
* Nutritional Status Prediction
* Streamlit Web Application
* Random Forest Classifier

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Machine Learning

## Model Performance

Accuracy: 93.8%

## Categories

* Normal
* Thinness
* Severe Thinness
* Overweight
* Obesity

## Run Locally

```bash
streamlit run app/app.py
```

## Streamlit Community Cloud

Streamlit Community Cloud requires this project to be in a public GitHub repository.

1. Create a Git repository in this folder.
2. Push the current branch to GitHub.
3. In Streamlit Cloud, connect the GitHub repo and set the app entry point to `app/app.py`.

Example commands:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
