
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# Task 1: Data Understanding
# --------------------------------------------------------------------------

DATA_PATH = "insurance.csv or your specific path for data"

def load_dataset(path=DATA_PATH, n_synthetic=1338):
  
    if os.path.exists(path):
        print(f"Loading real dataset from '{path}'")
        return pd.read_csv(path), False

    print(f"'{path}' not found locally — generating a synthetic dataset "
          f"with the same schema so the pipeline can run end-to-end.\n"
          f"Replace with the real Kaggle file before final submission.")

    age = np.random.randint(18, 65, n_synthetic)
    sex = np.random.choice(["male", "female"], n_synthetic)
    bmi = np.round(np.random.normal(30.7, 6.1, n_synthetic).clip(15, 54), 1)
    children = np.random.choice([0, 1, 2, 3, 4, 5],
                                n_synthetic, p=[0.43, 0.24, 0.18, 0.10, 0.03, 0.02])
    smoker = np.random.choice(["yes", "no"], n_synthetic, p=[0.205, 0.795])
    region = np.random.choice(
        ["southwest", "southeast", "northwest", "northeast"], n_synthetic
    )

    # Charges formula mirrors the known real-world relationships in this
    # dataset: base cost + age effect + bmi effect + a large smoker effect
    # + children effect + noise.
    charges = (
        250 * age
        + 300 * (bmi - 30)
        + 500 * children
        + (bmi > 30) * (smoker == "yes") * 20000  # obesity + smoking interaction
        + (smoker == "yes") * 15000
        + np.random.normal(0, 2000, n_synthetic)
        + 3000
    )
    charges = np.round(np.clip(charges, 1100, None), 2)

    df = pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region, "charges": charges
    })
    return df, True


df, is_synthetic = load_dataset()

print("\n=== First five records ===")
print(df.head())

print("\n=== Feature types ===")
target_variable = "charges"
numerical_features = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c != target_variable]
categorical_features = [c for c in df.select_dtypes(include=["object"]).columns]

print("Numerical features   :", numerical_features)
print("Categorical features :", categorical_features)
print("Target variable      :", target_variable)

# --------------------------------------------------------------------------
# Task 2: Data Preprocessing
# --------------------------------------------------------------------------

print("\n=== Missing values per column ===")
print(df.isnull().sum())

df_encoded = df.copy()
label_encoders = {}
for col in ["sex", "smoker", "region"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print("\n=== Encoding maps ===")
for col, mapping in label_encoders.items():
    print(f"{col}: {mapping}")

X = df_encoded[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df_encoded[target_variable]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size : {X_test.shape[0]} rows")

# --------------------------------------------------------------------------
# Task 3: Model Development
# --------------------------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Model coefficients ===")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:10s}: {coef:,.2f}")
print(f"{'intercept':10s}: {model.intercept_:,.2f}")

# --------------------------------------------------------------------------
# Task 4: Model Evaluation
# --------------------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluation metrics ===")
print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R2   : {r2:.4f}")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
print("\nSaved plot to actual_vs_predicted.png")

# Save metrics to a small text file for the README table
with open("metrics.txt", "w") as f:
    f.write(f"MAE,{mae:.2f}\nMSE,{mse:.2f}\nRMSE,{rmse:.2f}\nR2,{r2:.4f}\n")
    f.write(f"synthetic,{is_synthetic}\n")

print("\nDone.")
