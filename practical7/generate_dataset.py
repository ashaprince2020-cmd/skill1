import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "weight": np.round(np.random.uniform(45, 110, n), 1),
    "height": np.round(np.random.uniform(150, 195, n), 1),
    "exercise_hours": np.round(np.random.uniform(0, 10, n), 1),
    "sleep_hours": np.round(np.random.uniform(4, 10, n), 1),
    "sugar_intake": np.round(np.random.uniform(10, 150, n), 1),
    "smoking": np.random.randint(0, 2, n),
    "alcohol": np.random.randint(0, 2, n),
    "profession": np.random.randint(0, 4, n)
})

data["bmi"] = data["weight"] / ((data["height"] / 100) ** 2)

risk_score = (
    0.04 * data["age"]
    + 0.08 * data["bmi"]
    - 0.20 * data["exercise_hours"]
    - 0.15 * data["sleep_hours"]
    + 0.015 * data["sugar_intake"]
    + 0.8 * data["smoking"]
    + 0.5 * data["alcohol"]
)

probability = 1 / (1 + np.exp(-risk_score / 5))

data["risk"] = (np.random.random(n) < probability).astype(int)

data.to_csv("health_risk_dataset.csv", index=False)

print("Dataset created successfully")
print("Number of records:", len(data))
print(data.head())
