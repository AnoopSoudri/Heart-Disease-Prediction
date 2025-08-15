# src/features.py

import pandas as pd

def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example feature: Calculate BMI if weight and height columns exist.
    (Not in UCI dataset by default — placeholder example.)
    """
    if 'weight' in df.columns and 'height' in df.columns:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    return df

def categorize_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an 'age_group' categorical variable from 'age'.
    """
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'],
                                 bins=[0, 30, 45, 60, 120],
                                 labels=['<30', '30-45', '45-60', '60+'])
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering functions to the DataFrame.
    """
    df = add_bmi(df)
    df = categorize_age(df)
    return df
