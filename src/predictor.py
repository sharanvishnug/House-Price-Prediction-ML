"""
Prediction Engine

Loads the trained production model and predicts
house prices from user-provided features.
"""

import joblib
import pandas as pd
from pathlib import Path

# Project root directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"

MODEL_PATH = MODEL_DIR / "production_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Feature Order (Must Match Training)
# --------------------------------------------------

FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "house_age",
    "is_renovated",
    "renovation_age",
    "bath_bed_ratio",
    "sale_year",
    "sale_month",
]


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_house_price(features: dict) -> float:
    """
    Predict house price from input features.

    Parameters
    ----------
    features : dict
        Dictionary containing raw house features.

    Returns
    -------
    float
        Predicted house price.
    """

    bedrooms = features["bedrooms"]
    bathrooms = features["bathrooms"]
    yr_built = features["yr_built"]
    yr_renovated = features["yr_renovated"]
    sale_year = features["sale_year"]

    # -------------------------------
    # Derived Features
    # -------------------------------

    house_age = sale_year - yr_built

    is_renovated = 1 if yr_renovated > 0 else 0

    renovation_age = (
        sale_year - yr_renovated
        if yr_renovated > 0
        else 0
    )

    bath_bed_ratio = (
        bathrooms / bedrooms
        if bedrooms > 0
        else 0
    )

    # -------------------------------
    # Add derived features
    # -------------------------------

    features["house_age"] = house_age
    features["is_renovated"] = is_renovated
    features["renovation_age"] = renovation_age
    features["bath_bed_ratio"] = bath_bed_ratio

    # -------------------------------
    # Create DataFrame
    # -------------------------------

    input_df = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )

    # -------------------------------
    # Predict
    # -------------------------------

    prediction = model.predict(input_df)[0]

    return float(prediction)