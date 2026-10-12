"""
Model Evaluation and Visualization Module

This module calculates evaluation metrics:
- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1-Score (macro and weighted)
- Confusion Matrix
- Per-class Classification Report
- Feature Importance extraction

It also generates charts saved to assets/charts/.
"""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Visual theme setup: Academic Blue / Indigo / Slate
ACADEMIC_PRIMARY = "#2563EB"    # Royal Blue
ACADEMIC_SECONDARY = "#4F46E5"  # Indigo
ACADEMIC_ACCENT = "#0EA5E9"     # Sky Blue
ACADEMIC_DARK = "#1E293B"       # Slate 800
ACADEMIC_LIGHT = "#F8FAFC"      # Slate 50


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'precision_weighted': prec_weighted,
        'recall_macro': rec_macro,
        'recall_weighted': rec_weighted,
        'f1_macro': f1_mac,
        'f1_weighted': f1_wt,
        'confusion_matrix': cm,
        'classification_report_dict': report_dict,
        'classification_report_str': report_str,
        'labels': labels if labels else sorted(list(set(y_true)))
    }


def get_feature_importances(pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Extract and sort feature importances from a trained scikit-learn Pipeline.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    feature_names = preprocessor.get_feature_names_out()
    # Clean feature names (remove transformer prefixes like num__, cat__)
    cleaned_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

    importances = classifier.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': cleaned_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return df_imp


def plot_confusion_matrix(cm: List[List[int]], labels: List[str], save_path: str = None) -> plt.Figure:
    """
    Generate and save a confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=1.2,
        linecolor='white'
    )
    ax.set_title('Confusion Matrix: Student Performance Classification', fontsize=13, fontweight='bold', pad=12, color=ACADEMIC_DARK)
    ax.set_xlabel('Predicted Performance Tier', fontsize=11, fontweight='semibold', labelpad=8)
    ax.set_ylabel('True Performance Tier', fontsize=11, fontweight='semibold', labelpad=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def plot_feature_importance(df_imp: pd.DataFrame, top_n: int = 12, save_path: str = None) -> plt.Figure:
    """
    Generate and save a bar chart of top feature importances.
    """
    top_df = df_imp.head(top_n).iloc[::-1]  # reverse so highest is on top

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    bars = ax.barh(top_df['Feature'], top_df['Importance'], color=ACADEMIC_PRIMARY, edgecolor='none', height=0.65)

    # Add numeric labels at end of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va='center',
            ha='left',
            fontsize=9,
            color=ACADEMIC_DARK,
            fontweight='semibold'
        )

    ax.set_title(f'Top {top_n} Informative Features (Random Forest Gini Importance)', fontsize=12, fontweight='bold', pad=12, color=ACADEMIC_DARK)
    ax.set_xlabel('Mean Decrease in Impurity (Gini Importance)', fontsize=10, fontweight='semibold', labelpad=8)
    ax.set_ylabel('Feature Name', fontsize=10, fontweight='semibold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(top_df['Importance']) * 1.15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def plot_class_distribution(y_train: pd.Series, y_test: pd.Series, save_path: str = None) -> plt.Figure:
    """
    Visualize train and test target tier distribution.
    """
    df_dist = pd.DataFrame({
        'Train Set (80%)': y_train.value_counts(),
        'Test Set (20%)': y_test.value_counts()
    }).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    df_dist.plot(kind='bar', ax=ax, color=[ACADEMIC_PRIMARY, ACADEMIC_ACCENT], width=0.7, edgecolor='white')

    ax.set_title('Dataset Target Tier Stratification', fontsize=12, fontweight='bold', pad=12, color=ACADEMIC_DARK)
    ax.set_ylabel('Number of Students', fontsize=10, fontweight='semibold')
    ax.set_xlabel('Academic Performance Tier', fontsize=10, fontweight='semibold')
    ax.set_xticklabels(df_dist.index, rotation=0)
    ax.legend(frameon=True, facecolor=ACADEMIC_LIGHT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def plot_model_comparison(comparison_data: Dict[str, Dict[str, float]], save_path: str = None) -> plt.Figure:
    """
    Bar chart comparing Baseline Zero-R / Majority Classifier vs Random Forest Classifier.
    """
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    rf_vals = [
        comparison_data['Random Forest']['accuracy'],
        comparison_data['Random Forest']['f1_macro'],
        comparison_data['Random Forest']['f1_weighted']
    ]
    base_vals = [
        comparison_data['Baseline (Majority)']['accuracy'],
        comparison_data['Baseline (Majority)']['f1_macro'],
        comparison_data['Baseline (Majority)']['f1_weighted']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline (Majority Class)', color='#94A3B8')
    rects2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest Classifier', color=ACADEMIC_PRIMARY)

    # Label values
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.015, f'{h:.1%}', ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.015, f'{h:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Performance Comparison: Baseline vs Random Forest', fontsize=12, fontweight='bold', pad=12, color=ACADEMIC_DARK)
    ax.set_ylabel('Metric Score', fontsize=10, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def plot_system_architecture(save_path: str = None) -> plt.Figure:
    """
    Generate a clean visual flowchart diagram of the end-to-end ML architecture.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    ax.axis('off')

    # Draw boxes
    boxes = [
        {"text": "Educational\nDataset\n(UCI Repository\nN=395)", "x": 0.08, "y": 0.5, "w": 0.13, "h": 0.6, "color": "#E2E8F0"},
        {"text": "Data Preprocessing\n& Cleaning\n- Target Derivation\n- Leakage Exclusion\n- Imputation/Scaling", "x": 0.28, "y": 0.5, "w": 0.17, "h": 0.65, "color": "#DBEAFE"},
        {"text": "Stratified\nSplitting\n(80% Train / 20% Test)", "x": 0.50, "y": 0.5, "w": 0.14, "h": 0.5, "color": "#EDE9FE"},
        {"text": "Random Forest\nModel Training\n- 100 Estimators\n- Gini Criterion\n- Subsample Bagging", "x": 0.70, "y": 0.5, "w": 0.16, "h": 0.65, "color": "#FEF3C7"},
        {"text": "Evaluation &\nInteractive Web UI\n- Test Metrics\n- Streamlit Dashboard\n- Risk Assessment", "x": 0.90, "y": 0.5, "w": 0.15, "h": 0.65, "color": "#DCFCE7"}
    ]

    for b in boxes:
        rect = mpatches.FancyBboxPatch(
            (b["x"] - b["w"]/2, b["y"] - b["h"]/2),
            b["w"], b["h"],
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=b["color"],
            edgecolor=ACADEMIC_DARK,
            linewidth=1.4
        )
        ax.add_patch(rect)
        ax.text(
            b["x"], b["y"],
            b["text"],
            ha="center", va="center",
            fontsize=8.5,
            fontweight="semibold",
            color=ACADEMIC_DARK
        )

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x_start = boxes[i]["x"] + boxes[i]["w"]/2
        x_end = boxes[i+1]["x"] - boxes[i+1]["w"]/2
        ax.annotate(
            "",
            xy=(x_end, 0.5),
            xytext=(x_start, 0.5),
            arrowprops=dict(arrowstyle="->", lw=2, color=ACADEMIC_PRIMARY)
        )

    ax.set_title("End-to-End System Architecture: Machine Learning Pipeline", fontsize=13, fontweight="bold", pad=10, color=ACADEMIC_DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig
