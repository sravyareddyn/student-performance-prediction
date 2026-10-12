#!/usr/bin/env python3
"""
CLI Script to train, evaluate, and serialize the Student Performance Prediction model.

Usage:
    python train_model.py [--data data/student-mat.csv] [--seed 42]
"""

import argparse
import sys
import os
import json

from src.model_training import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest Classifier for Student Performance Prediction"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/student-mat.csv",
        help="Path to student dataset CSV file (default: data/student-mat.csv)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save model artifacts (default: models)"
    )
    parser.add_argument(
        "--charts-dir",
        type=str,
        default="assets/charts",
        help="Directory to save evaluation charts (default: assets/charts)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🎓 FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE: PROJECT PIPELINE")
    print("Project: Student Performance Prediction Using Machine Learning")
    print("Algorithm: Random Forest Classifier")
    print("=" * 70)

    if not os.path.exists(args.data):
        print(f"[-] Error: Dataset not found at {args.data}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[1/5] Loading and preprocessing dataset from '{args.data}'...")
    print(f"      Target: Derived academic tiers from final grade G3")
    print(f"      Leakage Protection: G3 & G2 strictly excluded from features")

    pipeline, results = train_and_evaluate(
        data_path=args.data,
        model_save_dir=args.models_dir,
        charts_save_dir=args.charts_dir,
        random_state=args.seed
    )

    rf_metrics = results['metrics']
    base_metrics = results['baseline_metrics']

    print(f"\n[2/5] Training Completed:")
    print(f"      Training samples: {results['train_samples']}")
    print(f"      Test samples:     {results['test_samples']}")
    print(f"      Classes:          {', '.join(results['target_classes'])}")

    print(f"\n[3/5] Empirical Evaluation on Unseen Test Set (20%):")
    print(f"      {'Metric':<25} {'Baseline (Majority)':<22} {'Random Forest':<15}")
    print(f"      {'-'*62}")
    print(f"      {'Accuracy':<25} {base_metrics['accuracy']:<22.4f} {rf_metrics['accuracy']:<15.4f}")
    print(f"      {'Macro F1-Score':<25} {base_metrics['f1_macro']:<22.4f} {rf_metrics['f1_macro']:<15.4f}")
    print(f"      {'Weighted F1-Score':<25} {base_metrics['f1_weighted']:<22.4f} {rf_metrics['f1_weighted']:<15.4f}")
    print(f"      {'Macro Precision':<25} {base_metrics['precision_macro']:<22.4f} {rf_metrics['precision_macro']:<15.4f}")
    print(f"      {'Macro Recall':<25} {base_metrics['recall_macro']:<22.4f} {rf_metrics['recall_macro']:<15.4f}")

    print("\n[4/5] Detailed Classification Report:")
    print(rf_metrics['classification_report_str'])

    print("[5/5] Artifacts and Visualizations Generated:")
    print(f"      - Trained Pipeline: {os.path.join(args.models_dir, 'student_rf_pipeline.joblib')}")
    print(f"      - Metrics Summary:  {os.path.join(args.models_dir, 'metrics.json')}")
    print(f"      - Confusion Matrix: {os.path.join(args.charts_dir, 'confusion_matrix.png')}")
    print(f"      - Feature Importances: {os.path.join(args.charts_dir, 'feature_importance.png')}")
    print(f"      - Class Distribution:  {os.path.join(args.charts_dir, 'class_distribution.png')}")
    print(f"      - Model Comparison:    {os.path.join(args.charts_dir, 'model_comparison.png')}")
    print(f"      - System Architecture: {os.path.join(args.charts_dir, 'system_architecture.png')}")
    print("\n✅ Model training and evaluation successfully executed!\n")


if __name__ == "__main__":
    main()
