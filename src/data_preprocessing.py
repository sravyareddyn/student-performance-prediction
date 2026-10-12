"""
Data Preprocessing Module for Student Performance Prediction

This module handles:
1. Loading the UCI Student Performance dataset.
2. Deriving the academic performance target category from the final grade (G3).
3. Explicit exclusion of intermediate/final grades (G2, G3) to prevent data leakage.
4. Robust imputing and encoding using scikit-learn ColumnTransformer.
5. Stratified train-test splitting.
"""

from typing import Tuple, List, Dict, Any
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Feature definitions for the model
# G3 is the source of the target variable and is strictly excluded from features.
# G2 (period 2 grade) is excluded to avoid late-stage trivial data leakage,
# ensuring the model can be used for early-warning interventions using G1 (baseline diagnostic).
NUMERICAL_FEATURES: List[str] = [
    'age',
    'Medu',
    'Fedu',
    'traveltime',
    'studytime',
    'failures',
    'famrel',
    'freetime',
    'goout',
    'Dalc',
    'Walc',
    'health',
    'absences',
    'G1'
]

CATEGORICAL_FEATURES: List[str] = [
    'sex',
    'address',
    'famsize',
    'Pstatus',
    'Mjob',
    'Fjob',
    'reason',
    'guardian',
    'schoolsup',
    'famsup',
    'paid',
    'activities',
    'nursery',
    'higher',
    'internet',
    'romantic'
]

ALL_MODEL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_FEATURES: List[str] = ['G3', 'G2', 'school']

TARGET_CLASSES: List[str] = [
    'Low (At-Risk)',
    'Medium (Satisfactory)',
    'High (Good)'
]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file. Handles both comma and semicolon separators.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")

    # The UCI student performance dataset uses semicolon as separator
    try:
        df = pd.read_csv(file_path, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep=',')
    except Exception:
        df = pd.read_csv(file_path, sep=',')

    return df


def map_grade_to_category(grade: float) -> str:
    """
    Categorize final grade (0-20 scale) into academic performance tiers:
    - Low (At-Risk): < 10 (official fail in Portuguese secondary system)
    - Medium (Satisfactory): 10 to 13 (sufficient pass)
    - High (Good): 14 to 20 (high achievement / distinction)
    """
    if pd.isna(grade):
        raise ValueError("Grade cannot be NaN")
    if grade < 10:
        return 'Low (At-Risk)'
    elif grade < 14:
        return 'Medium (Satisfactory)'
    else:
        return 'High (Good)'


def create_target(df: pd.DataFrame, grade_col: str = 'G3') -> pd.Series:
    """
    Derive the categorical target from the continuous grade column.
    """
    if grade_col not in df.columns:
        raise KeyError(f"Target source column '{grade_col}' not present in dataset.")
    return df[grade_col].apply(map_grade_to_category)


def build_preprocessor(
    num_features: List[str] = None,
    cat_features: List[str] = None
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for preprocessing:
    - Numerical: median imputation + standard scaling
    - Categorical: most frequent imputation + OneHotEncoding (drop='first', handle_unknown='ignore')
    """
    if num_features is None:
        num_features = NUMERICAL_FEATURES
    if cat_features is None:
        cat_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop'
    )

    return preprocessor


def prepare_data(
    file_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Execute full preprocessing pipeline:
    1. Load raw data
    2. Construct performance target
    3. Extract input feature matrix without data leakage
    4. Stratified train-test split
    """
    df = load_data(file_path)
    y = create_target(df, grade_col='G3')

    # Ensure all required features are present
    available_features = [col for col in ALL_MODEL_FEATURES if col in df.columns]
    X = df[available_features].copy()

    # Stratified train-test split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    available_num = [f for f in NUMERICAL_FEATURES if f in available_features]
    available_cat = [f for f in CATEGORICAL_FEATURES if f in available_features]
    preprocessor = build_preprocessor(available_num, available_cat)

    return X_train, X_test, y_train, y_test, preprocessor
