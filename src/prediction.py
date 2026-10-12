"""
Inference and Prediction Module for Student Performance

This module:
1. Loads the saved Joblib pipeline.
2. Validates incoming student feature inputs.
3. Generates category predictions and calibrated class probabilities.
4. Generates contextual academic recommendations based on input factors.
5. Emphasizes responsible AI disclaimers and human-in-the-loop decision making.
"""

from typing import Dict, Any, List, Union
import os
import joblib
import pandas as pd
import numpy as np

from src.data_preprocessing import ALL_MODEL_FEATURES, TARGET_CLASSES


DEFAULT_STUDENT_PROFILE: Dict[str, Any] = {
    'age': 16,
    'sex': 'F',
    'address': 'U',
    'famsize': 'GT3',
    'Pstatus': 'T',
    'Medu': 2,
    'Fedu': 2,
    'Mjob': 'other',
    'Fjob': 'other',
    'reason': 'course',
    'guardian': 'mother',
    'traveltime': 1,
    'studytime': 2,
    'failures': 0,
    'schoolsup': 'no',
    'famsup': 'no',
    'paid': 'no',
    'activities': 'no',
    'nursery': 'yes',
    'higher': 'yes',
    'internet': 'yes',
    'romantic': 'no',
    'famrel': 4,
    'freetime': 3,
    'goout': 3,
    'Dalc': 1,
    'Walc': 1,
    'health': 3,
    'absences': 4,
    'G1': 10.0
}


class StudentPerformancePredictor:
    """
    Predictor wrapper around the trained scikit-learn Pipeline.
    """

    def __init__(self, model_path: str = 'models/student_rf_pipeline.joblib'):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                "Please run 'python train_model.py' first."
            )
        self.model = joblib.load(self.model_path)

    def validate_inputs(self, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate, sanitize, and fill missing features using defaults.
        """
        sanitized = dict(DEFAULT_STUDENT_PROFILE)
        sanitized.update(student_dict)

        # Range checks
        if 'age' in sanitized:
            sanitized['age'] = max(15, min(25, int(sanitized['age'])))
        if 'G1' in sanitized:
            sanitized['G1'] = max(0.0, min(20.0, float(sanitized['G1'])))
        if 'absences' in sanitized:
            sanitized['absences'] = max(0, min(93, int(sanitized['absences'])))
        if 'studytime' in sanitized:
            sanitized['studytime'] = max(1, min(4, int(sanitized['studytime'])))
        if 'failures' in sanitized:
            sanitized['failures'] = max(0, min(4, int(sanitized['failures'])))

        return sanitized

    def generate_recommendations(self, student_dict: Dict[str, Any], predicted_tier: str) -> List[str]:
        """
        Generate actionable pedagogical suggestions based on risk factors.
        """
        recs = []

        # Study time
        studytime = int(student_dict.get('studytime', 2))
        if studytime <= 2:
            recs.append("Increase weekly structured study time to at least 5-10 hours with scheduled revision blocks.")

        # Absences
        absences = int(student_dict.get('absences', 0))
        if absences > 10:
            recs.append(f"Student has high absenteeism ({absences} days). Prioritize attendance monitoring and identify underlying barriers.")

        # Failures
        failures = int(student_dict.get('failures', 0))
        if failures > 0:
            recs.append(f"Address persistent learning gaps in foundational prerequisite concepts ({failures} past failures detected).")

        # Baseline diagnostic (G1)
        g1 = float(student_dict.get('G1', 10.0))
        if g1 < 10.0:
            recs.append("Early assessment score (G1 < 10) indicates critical risk. Provide immediate remedial tutoring and weekly formative check-ins.")
        elif g1 < 14.0:
            recs.append("Moderate early assessment score. Offer guided practice sessions to transition understanding into mastery.")

        # Educational Support
        if student_dict.get('schoolsup') == 'no' and predicted_tier == 'Low (At-Risk)':
            recs.append("Enroll student in institutional extra educational support / peer tutoring programs.")

        if not recs:
            recs.append("Maintain consistent study habits, active classroom engagement, and continuous self-assessment.")

        return recs

    def predict(self, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction, probabilities, confidence, recommendations, and disclaimer.
        """
        if self.model is None:
            self._load_model()

        validated = self.validate_inputs(student_dict)
        # Ensure DataFrame has exact feature columns in order
        df_input = pd.DataFrame([validated])[ALL_MODEL_FEATURES]

        # Predict tier
        predicted_tier = str(self.model.predict(df_input)[0])

        # Predict class probabilities
        probs = self.model.predict_proba(df_input)[0]
        classes = self.model.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

        confidence = float(np.max(probs))

        # Risk indicator
        if predicted_tier == 'Low (At-Risk)':
            risk_status = 'High Academic Risk'
            badge_color = 'red'
        elif predicted_tier == 'Medium (Satisfactory)':
            risk_status = 'Moderate / Stable'
            badge_color = 'amber'
        else:
            risk_status = 'High Performance'
            badge_color = 'green'

        recommendations = self.generate_recommendations(validated, predicted_tier)

        disclaimer = (
            "IMPORTANT NOTICE: Predictions generated by this machine learning system are "
            "probabilistic estimates based on historical student population patterns. "
            "They DO NOT represent a definitive judgment on an individual student's potential, "
            "intelligence, or future academic outcome. This tool must NOT be used for automated, "
            "punitive, or consequential administrative decisions. It is designed solely as a "
            "support aid for educators and academic advisors to identify early opportunities for "
            "constructive intervention."
        )

        return {
            'predicted_category': predicted_tier,
            'confidence': confidence,
            'probabilities': prob_dict,
            'risk_status': risk_status,
            'badge_color': badge_color,
            'recommendations': recommendations,
            'disclaimer': disclaimer,
            'input_data': validated
        }

    def predict_batch(self, df_inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for multiple records at once, filling missing columns from defaults.
        """
        if self.model is None:
            self._load_model()

        # Fill missing columns from default profile
        df_full = df_inputs.copy()
        for col, default_val in DEFAULT_STUDENT_PROFILE.items():
            if col not in df_full.columns:
                df_full[col] = default_val

        df_ordered = df_full[ALL_MODEL_FEATURES]

        preds = self.model.predict(df_ordered)
        probs = self.model.predict_proba(df_ordered)
        classes = self.model.classes_

        df_out = df_inputs.copy()
        df_out['Predicted_Category'] = preds
        for i, cls in enumerate(classes):
            df_out[f'Probability_{cls}'] = probs[:, i]

        return df_out
