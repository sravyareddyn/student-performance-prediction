"""
Unit tests for model inference, prediction validation, and output formats.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.prediction import StudentPerformancePredictor
from src.data_preprocessing import TARGET_CLASSES


@pytest.fixture(scope="module")
def predictor():
    model_path = 'models/student_rf_pipeline.joblib'
    if not os.path.exists(model_path):
        pytest.fail(f"Trained model not found at {model_path}. Train before testing.")
    return StudentPerformancePredictor(model_path=model_path)


def test_model_loading(predictor):
    assert predictor.model is not None


def test_valid_student_prediction(predictor):
    # Typical student profile
    sample_student = {
        'age': 16,
        'sex': 'F',
        'address': 'U',
        'famsize': 'GT3',
        'Pstatus': 'T',
        'Medu': 3,
        'Fedu': 3,
        'Mjob': 'services',
        'Fjob': 'other',
        'reason': 'reputation',
        'guardian': 'mother',
        'traveltime': 1,
        'studytime': 2,
        'failures': 0,
        'schoolsup': 'no',
        'famsup': 'yes',
        'paid': 'no',
        'activities': 'yes',
        'nursery': 'yes',
        'higher': 'yes',
        'internet': 'yes',
        'romantic': 'no',
        'famrel': 4,
        'freetime': 3,
        'goout': 3,
        'Dalc': 1,
        'Walc': 1,
        'health': 4,
        'absences': 2,
        'G1': 14.0
    }

    result = predictor.predict(sample_student)

    assert 'predicted_category' in result
    assert result['predicted_category'] in TARGET_CLASSES
    assert 'confidence' in result
    assert 0.0 <= result['confidence'] <= 1.0
    assert 'probabilities' in result
    probs = result['probabilities']
    assert set(probs.keys()) == set(TARGET_CLASSES)
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0
    assert 'risk_status' in result
    assert 'recommendations' in result
    assert isinstance(result['recommendations'], list)
    assert len(result['recommendations']) > 0
    assert 'disclaimer' in result
    assert "not be used for automated" in result['disclaimer'].lower()


def test_high_risk_student_prediction(predictor):
    # High risk student profile (multiple failures, low G1, high absences)
    at_risk_student = {
        'age': 18,
        'sex': 'M',
        'address': 'R',
        'famsize': 'GT3',
        'Pstatus': 'A',
        'Medu': 1,
        'Fedu': 1,
        'Mjob': 'other',
        'Fjob': 'other',
        'reason': 'course',
        'guardian': 'other',
        'traveltime': 3,
        'studytime': 1,
        'failures': 3,
        'schoolsup': 'no',
        'famsup': 'no',
        'paid': 'no',
        'activities': 'no',
        'nursery': 'no',
        'higher': 'no',
        'internet': 'no',
        'romantic': 'yes',
        'famrel': 2,
        'freetime': 5,
        'goout': 5,
        'Dalc': 4,
        'Walc': 5,
        'health': 2,
        'absences': 22,
        'G1': 4.0
    }

    result = predictor.predict(at_risk_student)
    assert result['predicted_category'] in TARGET_CLASSES
    assert result['probabilities']['Low (At-Risk)'] > 0.40


def test_input_validation_clamping(predictor):
    # Test extreme out-of-range inputs
    extreme_student = {
        'age': 99,           # Out of realistic bounds
        'absences': 500,     # Out of realistic bounds
        'G1': 35.0,          # Out of 0-20 scale
        'studytime': -5,     # Below minimum
        'failures': 10       # Above max
    }

    sanitized = predictor.validate_inputs(extreme_student)
    assert sanitized['age'] == 25
    assert sanitized['absences'] == 93
    assert sanitized['G1'] == 20.0
    assert sanitized['studytime'] == 1
    assert sanitized['failures'] == 4


def test_missing_features_handled_gracefully(predictor):
    # Minimal inputs: missing features should be imputed without crashing
    minimal_student = {
        'age': 16,
        'G1': 11.0,
        'studytime': 2
    }

    result = predictor.predict(minimal_student)
    assert result['predicted_category'] in TARGET_CLASSES


def test_batch_prediction(predictor):
    df_batch = pd.DataFrame([
        {'age': 16, 'studytime': 3, 'failures': 0, 'G1': 15.0, 'absences': 1},
        {'age': 17, 'studytime': 1, 'failures': 2, 'G1': 6.0, 'absences': 14}
    ])

    df_out = predictor.predict_batch(df_batch)
    assert 'Predicted_Category' in df_out.columns
    assert len(df_out) == 2
    assert all(cat in TARGET_CLASSES for cat in df_out['Predicted_Category'])
