"""
Unit tests for data preprocessing and pipeline transformations.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    load_data,
    map_grade_to_category,
    create_target,
    build_preprocessor,
    prepare_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_CLASSES
)


def test_map_grade_to_category():
    assert map_grade_to_category(0.0) == 'Low (At-Risk)'
    assert map_grade_to_category(9.0) == 'Low (At-Risk)'
    assert map_grade_to_category(9.9) == 'Low (At-Risk)'
    assert map_grade_to_category(10.0) == 'Medium (Satisfactory)'
    assert map_grade_to_category(13.9) == 'Medium (Satisfactory)'
    assert map_grade_to_category(14.0) == 'High (Good)'
    assert map_grade_to_category(20.0) == 'High (Good)'

    with pytest.raises(ValueError):
        map_grade_to_category(np.nan)


def test_load_data_valid():
    df = load_data('data/student-mat.csv')
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 395
    assert 'G3' in df.columns
    assert 'G1' in df.columns


def test_load_data_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_data('non_existent_file.csv')


def test_create_target():
    df = pd.DataFrame({'G3': [5, 11, 15, 8, 13, 18]})
    target = create_target(df)
    assert len(target) == 6
    assert target.iloc[0] == 'Low (At-Risk)'
    assert target.iloc[1] == 'Medium (Satisfactory)'
    assert target.iloc[2] == 'High (Good)'


def test_missing_value_handling_in_preprocessor():
    """
    Verify that the ColumnTransformer handles NaN / missing values properly
    using SimpleImputer for both numeric (median) and categorical (mode).
    """
    preprocessor = build_preprocessor()

    # Synthetic sample with missing values
    data = {
        'age': [16, np.nan, 17],
        'Medu': [4, 2, np.nan],
        'Fedu': [np.nan, 3, 2],
        'traveltime': [1, 2, 1],
        'studytime': [2, np.nan, 3],
        'failures': [0, 1, np.nan],
        'famrel': [4, 5, 3],
        'freetime': [3, np.nan, 4],
        'goout': [2, 3, np.nan],
        'Dalc': [1, 1, 2],
        'Walc': [1, 2, np.nan],
        'health': [5, np.nan, 4],
        'absences': [2, 4, np.nan],
        'G1': [12.0, np.nan, 15.0],
        'sex': ['F', None, 'M'],
        'address': ['U', 'R', None],
        'famsize': ['GT3', None, 'LE3'],
        'Pstatus': ['T', 'A', None],
        'Mjob': ['teacher', None, 'services'],
        'Fjob': [None, 'health', 'other'],
        'reason': ['home', 'course', None],
        'guardian': ['mother', None, 'father'],
        'schoolsup': ['yes', 'no', None],
        'famsup': [None, 'yes', 'no'],
        'paid': ['no', 'yes', None],
        'activities': ['yes', None, 'no'],
        'nursery': ['yes', 'yes', None],
        'higher': ['yes', None, 'yes'],
        'internet': ['yes', 'no', None],
        'romantic': [None, 'no', 'yes']
    }
    df_with_nans = pd.DataFrame(data)

    # Fitting preprocessor on DataFrame with missing values should succeed without error
    transformed = preprocessor.fit_transform(df_with_nans)
    assert transformed is not None
    assert not np.isnan(transformed).any()


def test_stratified_split_preserves_classes():
    X_train, X_test, y_train, y_test, preprocessor = prepare_data('data/student-mat.csv')
    assert len(X_train) + len(X_test) == 395
    assert set(y_train.unique()) == set(TARGET_CLASSES)
    assert set(y_test.unique()) == set(TARGET_CLASSES)
    # Check that neither G2 nor G3 are in X_train or X_test (leakage prevention)
    assert 'G3' not in X_train.columns
    assert 'G2' not in X_train.columns
    assert 'G3' not in X_test.columns
    assert 'G2' not in X_test.columns
