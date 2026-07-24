import pandas as pd
from datetime import datetime


CURRENT_YEAR = datetime.now().year


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.
    """

    df = df.copy()

    # -------------------------------
    # House Age
    # -------------------------------
    df["house_age"] = CURRENT_YEAR - df["yr_built"]

    # -------------------------------
    # Renovated?
    # -------------------------------
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)

    # -------------------------------
    # Renovation Age
    # -------------------------------
    df["renovation_age"] = df["yr_renovated"].apply(
        lambda year: CURRENT_YEAR - year if year > 0 else 0
    )

    # -------------------------------
    # Bathroom / Bedroom Ratio
    # -------------------------------
    df["bath_bed_ratio"] = (
        df["bathrooms"] / df["bedrooms"].replace(0, 1)
    )

    return df