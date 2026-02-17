import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv("kidney_disease.csv")

# Drop id if exists
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Clean target column
df["classification"] = df["classification"].astype(str).str.strip().str.lower()
df["classification"] = df["classification"].replace({"ckd": 1, "notckd": 0})

# Fix common dirty labels like "ckd\t"
df["classification"] = df["classification"].replace({"ckd\t": 1, "notckd\t": 0})

# ------------------------
# Features
# ------------------------
FEATURE_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc"
]

CATEGORICAL_COLS = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad",
    "appet", "pe", "ane"
]

df = df[FEATURE_COLUMNS + ["classification"]]

# Clean categorical columns
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace(["nan", "none", ""], np.nan)

# Convert numeric columns
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

X = df[FEATURE_COLUMNS]
y = df["classification"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------
# Impute
# ------------------------
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train_num = num_imputer.fit_transform(X_train[NUMERIC_COLS])
X_test_num = num_imputer.transform(X_test[NUMERIC_COLS])

X_train_cat = cat_imputer.fit_transform(X_train[CATEGORICAL_COLS])
X_test_cat = cat_imputer.transform(X_test[CATEGORICAL_COLS])

X_train_num = pd.DataFrame(X_train_num, columns=NUMERIC_COLS)
X_test_num = pd.DataFrame(X_test_num, columns=NUMERIC_COLS)

X_train_cat = pd.DataFrame(X_train_cat, columns=CATEGORICAL_COLS)
X_test_cat = pd.DataFrame(X_test_cat, columns=CATEGORICAL_COLS)

# ------------------------
# Label Encode categorical
# ------------------------
label_encoders = {}

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    X_train_cat[col] = le.fit_transform(X_train_cat[col].astype(str))
    X_test_cat[col] = X_test_cat[col].astype(str).apply(
        lambda x: x if x in le.classes_ else le.classes_[0]
    )
    X_test_cat[col] = le.transform(X_test_cat[col])
    label_encoders[col] = le

X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

# ------------------------
# Train model
# ------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_final, y_train)

pred = model.predict(X_test_final)
acc = accuracy_score(y_test, pred)

print("Test Accuracy:", acc)

# ------------------------
# Save
# ------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/ckd_xgboost_model.pkl")
joblib.dump(num_imputer, "models/num_imputer.pkl")
joblib.dump(cat_imputer, "models/cat_imputer.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("Saved all models into /models folder")
