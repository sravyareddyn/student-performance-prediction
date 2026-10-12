"""
Model Training Module for Student Performance Prediction

This module handles:
1. Combining the preprocessor and Random Forest Classifier into a scikit-learn Pipeline.
2. Fitting the baseline (majority class) model for benchmark comparison.
3. Fitting the Random Forest Classifier on training data.
4. Evaluating on unseen test data.
5. Serializing model pipeline and metadata using Joblib and JSON.
"""

from typing import Dict, Any, Tuple
import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer

from src.data_preprocessing import (
    prepare_data,
    TARGET_CLASSES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_MODEL_FEATURES
)
from src.evaluation import (
    compute_metrics,
    get_feature_importances,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_class_distribution,
    plot_model_comparison,
    plot_system_architecture
)


def create_rf_pipeline(
    preprocessor: ColumnTransformer,
    n_estimators: int = 100,
    max_depth: int = 6,
    min_samples_split: int = 4,
    random_state: int = 42
) -> Pipeline:
    """
    Build a complete Pipeline containing ColumnTransformer and RandomForestClassifier.
    """
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])
    return pipeline


def train_and_evaluate(
    data_path: str = 'data/student-mat.csv',
    model_save_dir: str = 'models',
    charts_save_dir: str = 'assets/charts',
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    End-to-end training and evaluation:
    - Preprocesses data
    - Fits Baseline & Random Forest
    - Computes real test metrics
    - Saves model and artifacts
    - Generates all evaluation figures
    """
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(charts_save_dir, exist_ok=True)

    # 1. Prepare data (Stratified Split)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        data_path, test_size=0.2, random_state=random_state
    )

    # 2. Train Baseline (Majority Class)
    baseline_clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DummyClassifier(strategy='most_frequent'))
    ])
    baseline_clf.fit(X_train, y_train)
    y_pred_baseline = baseline_clf.predict(X_test)
    baseline_metrics = compute_metrics(y_test, y_pred_baseline, labels=TARGET_CLASSES)

    # 3. Build & Train Random Forest Pipeline
    rf_pipeline = create_rf_pipeline(
        preprocessor,
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        random_state=random_state
    )
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    rf_metrics = compute_metrics(y_test, y_pred_rf, labels=TARGET_CLASSES)

    # 4. Extract Feature Importances
    df_importances = get_feature_importances(rf_pipeline, top_n=25)

    # 5. Compile Results Dictionary
    results: Dict[str, Any] = {
        'model_name': 'Random Forest Classifier',
        'dataset': 'UCI Student Performance (Math)',
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'random_state': random_state,
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 4,
            'criterion': 'gini'
        },
        'metrics': rf_metrics,
        'baseline_metrics': baseline_metrics,
        'features_used': ALL_MODEL_FEATURES,
        'target_classes': TARGET_CLASSES
    }

    # 6. Save Model and Metadata
    model_path = os.path.join(model_save_dir, 'student_rf_pipeline.joblib')
    joblib.dump(rf_pipeline, model_path)

    metrics_path = os.path.join(model_save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)

    importances_path = os.path.join(model_save_dir, 'feature_importances.csv')
    df_importances.to_csv(importances_path, index=False)

    # 7. Generate Evaluation Visualizations
    cm_path = os.path.join(charts_save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(rf_metrics['confusion_matrix'], TARGET_CLASSES, save_path=cm_path)

    fi_path = os.path.join(charts_save_dir, 'feature_importance.png')
    plot_feature_importance(df_importances, top_n=12, save_path=fi_path)

    dist_path = os.path.join(charts_save_dir, 'class_distribution.png')
    plot_class_distribution(y_train, y_test, save_path=dist_path)

    comp_path = os.path.join(charts_save_dir, 'model_comparison.png')
    comparison_data = {
        'Baseline (Majority)': baseline_metrics,
        'Random Forest': rf_metrics
    }
    plot_model_comparison(comparison_data, save_path=comp_path)

    arch_path = os.path.join(charts_save_dir, 'system_architecture.png')
    plot_system_architecture(save_path=arch_path)

    return rf_pipeline, results
