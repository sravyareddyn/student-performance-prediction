# Models Directory: Serialized Artifacts & Metadata

**Project:** Student Performance Prediction Using Machine Learning  
**Algorithm:** Random Forest Classifier  
**Subject:** Fundamentals of Artificial Intelligence  

---

## 1. Directory Contents

| Artifact File | Size | Description |
| :--- | :--- | :--- |
| `student_rf_pipeline.joblib` | ~1.4 MB | Serialized end-to-end Scikit-learn Pipeline (`ColumnTransformer` + `RandomForestClassifier`) |
| `metrics.json` | ~4 KB | Machine-readable empirical evaluation metrics on unseen held-out test data (N=79) |
| `feature_importances.csv` | ~1.2 KB | Ranked Gini feature importances across all 41 one-hot encoded input dimensions |

---

## 2. Pipeline Architecture & Components

The saved pipeline integrates preprocessing and classification into a unified, leak-free object:

```
Raw Input Data (24 Features)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│                   ColumnTransformer                    │
├────────────────────────────┬───────────────────────────┤
│    Numerical Pipeline      │   Categorical Pipeline    │
│  - SimpleImputer (median)  │  - SimpleImputer (mode)   │
│  - StandardScaler          │  - OneHotEncoder (drop='first',
│    (14 features)           │       handle_unknown='ignore')
│                            │    (10 features)          │
└────────────────────────────┴───────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│             Random Forest Classifier                   │
│  - n_estimators: 100                                   │
│  - max_depth: 6                                        │
│  - min_samples_split: 4                                │
│  - criterion: 'gini'                                   │
│  - random_state: 42                                    │
└────────────────────────────────────────────────────────┘
            │
            ▼
Prediction: ['Low (At-Risk)', 'Medium (Satisfactory)', 'High (Good)']
+ Calibrated Class Probabilities
```

---

## 3. Empirical Test Metrics (Held-Out Test Set N = 79)

| Metric | Majority Baseline (Zero-R) | Random Forest Classifier | Absolute Gain |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.4177 (41.77%) | **0.6709 (67.09%)** | **+25.32%** |
| **Macro F1-Score** | 0.1964 | **0.6845** | **+0.4881** |
| **Weighted F1-Score** | 0.2462 | **0.6724** | **+0.4262** |
| **Macro Precision** | 0.1392 | **0.7016** | **+0.5624** |
| **Macro Recall** | 0.3333 | **0.6746** | **+0.3413** |

### Per-Class Performance Breakdown:
- **Low (At-Risk)**: Precision: `0.6818`, Recall: `0.5769`, F1-Score: `0.6250` (Support: 26)
- **Medium (Satisfactory)**: Precision: `0.5897`, Recall: `0.6970`, F1-Score: `0.6389` (Support: 33)
- **High (Good)**: Precision: `0.8261`, Recall: `0.7500`, F1-Score: `0.7895` (Support: 20)

---

## 4. How to Load and Execute Inference

```python
import joblib
import pandas as pd

# 1. Load pipeline from disk
pipeline = joblib.load('models/student_rf_pipeline.joblib')

# 2. Provide student input dictionary (24 features)
student_data = {
    'age': 16, 'sex': 'F', 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',
    'Medu': 3, 'Fedu': 3, 'Mjob': 'services', 'Fjob': 'other',
    'reason': 'reputation', 'guardian': 'mother', 'traveltime': 1,
    'studytime': 3, 'failures': 0, 'schoolsup': 'no', 'famsup': 'yes',
    'paid': 'no', 'activities': 'yes', 'nursery': 'yes', 'higher': 'yes',
    'internet': 'yes', 'romantic': 'no', 'famrel': 4, 'freetime': 3,
    'goout': 2, 'Dalc': 1, 'Walc': 1, 'health': 4, 'absences': 2,
    'G1': 15.0
}
df_input = pd.DataFrame([student_data])

# 3. Generate prediction
predicted_tier = pipeline.predict(df_input)[0]
probabilities = pipeline.predict_proba(df_input)[0]

print(f"Predicted Academic Tier: {predicted_tier}")
for cls, prob in zip(pipeline.classes_, probabilities):
    print(f"  P({cls}): {prob:.1%}")
```
