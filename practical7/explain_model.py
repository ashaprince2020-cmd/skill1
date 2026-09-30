import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

data = pd.read_csv("health_risk_dataset.csv")

X = data.drop("risk", axis=1)

model = joblib.load("xgboost.pkl")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig("shap_summary.png", dpi=300)

plt.close()

print("SHAP analysis completed successfully")
print("SHAP summary plot saved as shap_summary.png")
