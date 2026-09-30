import pandas as pd
import joblib

model = joblib.load("xgboost.pkl")

print("Health Risk Prediction Demo")
print("Enter the following details:")

age = float(input("Age: "))
weight = float(input("Weight (kg): "))
height = float(input("Height (cm): "))
exercise = float(input("Exercise hours per week: "))
sleep = float(input("Sleep hours per day: "))
sugar = float(input("Sugar intake: "))
smoking = int(input("Smoking (0=No, 1=Yes): "))
alcohol = int(input("Alcohol (0=No, 1=Yes): "))
profession = int(input("Profession (0-3): "))

bmi = weight / ((height / 100) ** 2)

input_data = pd.DataFrame([{
    "age": age,
    "weight": weight,
    "height": height,
    "exercise_hours": exercise,
    "sleep_hours": sleep,
    "sugar_intake": sugar,
    "smoking": smoking,
    "alcohol": alcohol,
    "profession": profession,
    "bmi": bmi
}])

prediction = model.predict(input_data)[0]

if prediction == 1:
    print("\nPredicted Risk: HIGH")
else:
    print("\nPredicted Risk: LOW")

print("\nNote: This is an educational prediction using synthetic data, not a medical diagnosis.")
