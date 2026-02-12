from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load saved model + preprocessors
# -------------------------------
model = joblib.load("models/ckd_xgboost_model.pkl")
num_imputer = joblib.load("models/num_imputer.pkl")
cat_imputer = joblib.load("models/cat_imputer.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Column order must match training dataset
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

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="CKD Prediction API", version="1.0")


# -------------------------------
# Input Schema (Pydantic)
# -------------------------------
class CKDInput(BaseModel):
    age: float | None = None
    bp: float | None = None
    sg: float | None = None
    al: float | None = None
    su: float | None = None

    rbc: str | None = None
    pc: str | None = None
    pcc: str | None = None
    ba: str | None = None

    bgr: float | None = None
    bu: float | None = None
    sc: float | None = None
    sod: float | None = None
    pot: float | None = None
    hemo: float | None = None
    pcv: float | None = None
    wc: float | None = None
    rc: float | None = None

    htn: str | None = None
    dm: str | None = None
    cad: str | None = None
    appet: str | None = None
    pe: str | None = None
    ane: str | None = None


# -------------------------------
# Helper: preprocess
# -------------------------------
def preprocess_input(data: dict):
    # Convert input dict -> DataFrame
    df = pd.DataFrame([data])

    # Ensure all required columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Keep correct order
    df = df[FEATURE_COLUMNS]

    # Strip spaces from categorical text
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("None", np.nan)
        df[col] = df[col].replace("nan", np.nan)

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Split for imputation
    df_num = df[NUMERIC_COLS]
    df_cat = df[CATEGORICAL_COLS]

    # Impute
    df_num = pd.DataFrame(num_imputer.transform(df_num), columns=NUMERIC_COLS)
    df_cat = pd.DataFrame(cat_imputer.transform(df_cat), columns=CATEGORICAL_COLS)

    # Encode categorical columns using saved LabelEncoders
    for col in CATEGORICAL_COLS:
        le = label_encoders[col]

        # Handle unknown category safely
        df_cat[col] = df_cat[col].apply(lambda x: x if x in le.classes_ else "missing")

        # If "missing" not present in encoder, fallback
        if "missing" not in le.classes_:
            # replace unknown with first class
            df_cat[col] = df_cat[col].apply(lambda x: le.classes_[0] if x not in le.classes_ else x)

        df_cat[col] = le.transform(df_cat[col])

    # Combine
    final_df = pd.concat([df_num, df_cat], axis=1)

    # Reorder exactly like training
    final_df = final_df[NUMERIC_COLS + CATEGORICAL_COLS]

    return final_df


# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "CKD Prediction API is running"}


@app.post("/predict")
def predict_ckd(input_data: CKDInput):
    data_dict = input_data.dict()

    processed = preprocess_input(data_dict)

    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]

    return {
        "prediction": "CKD" if pred == 1 else "NOT CKD",
        "ckd_probability": float(prob)
    }
