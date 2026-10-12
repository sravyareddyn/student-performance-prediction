"""
Student Performance Prediction Dashboard
Fundamentals of Artificial Intelligence - Academic Project
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.prediction import StudentPerformancePredictor, DEFAULT_STUDENT_PROFILE
from src.data_preprocessing import load_data, create_target, TARGET_CLASSES, ALL_MODEL_FEATURES

# Configure Streamlit page
st.set_page_config(
    page_title="EduPredict AI | Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Academic Theme (Blue, Indigo, Slate, Crisp White)
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #F8FAFC;
    }
    
    /* Academic Header */
    .academic-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #4F46E5 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .academic-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .academic-header p {
        color: #E0E7FF !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Metric Card */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-top: 0.25rem;
    }
    
    /* Prediction Badges */
    .badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #31C48D;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF08A;
        color: #713F12;
        border: 1px solid #FACC15;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F98080;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
        font-size: 0.92rem;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Load the trained model pipeline once and cache in memory."""
    model_path = 'models/student_rf_pipeline.joblib'
    if not os.path.exists(model_path):
        st.error(f"Model artifact not found at '{model_path}'. Please run 'python train_model.py' first.")
        st.stop()
    return StudentPerformancePredictor(model_path=model_path)


@st.cache_data
def get_metrics_metadata():
    """Load precomputed metrics and evaluation artifacts."""
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def get_raw_dataset():
    """Load dataset for exploration."""
    return load_data('data/student-mat.csv')


# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=64)
st.sidebar.title("EduPredict AI")
st.sidebar.caption("Fundamentals of AI — Project Portal")

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🎯 Predict Performance",
        "📊 Dataset Explorer",
        "📈 Model Evaluation",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Specification:**
- Algorithm: **Random Forest**
- Trees: **100**
- Dataset: **UCI Math (N=395)**
- Input Features: **24 Variables**
- Target: **3 Academic Tiers**
""")


# ==========================================
# PAGE 1: HOME / OVERVIEW
# ==========================================
if selected_page == "🏠 Overview":
    st.markdown("""
    <div class="academic-header">
        <h1>Student Performance Prediction System</h1>
        <p>A Machine Learning Framework for Proactive Educational Intervention & Academic Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">67.1%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.685</div>
            <div class="metric-label">Macro F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">+25.3%</div>
            <div class="metric-label">Over Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Zero</div>
            <div class="metric-label">Data Leakage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📌 Executive Summary")
    st.markdown("""
    Educational institutions collect rich data regarding student engagement, family background, 
    behavioral patterns, and early diagnostic assessments. Often, students falling at risk of academic 
    underachievement are identified only after summative final examinations, when timely remedial support 
    is no longer possible.

    This project implements an empirical **Machine Learning decision-support tool** powered by an 
    optimized **Random Forest Classifier**. By analyzing early diagnostic assessments (Period 1 test scores) 
    alongside socio-academic factors (study habits, past course failures, absenteeism, and parental support), 
    the system categorizes students into three strategic academic performance tiers:
    1. **Low (At-Risk)**: Final score < 10 (official threshold for course failure).
    2. **Medium (Satisfactory)**: Final score 10–13 (sufficient passing competency).
    3. **High (Good / Distinction)**: Final score 14–20 (upper academic distinction).
    """)

    col_arch1, col_arch2 = st.columns([1.1, 0.9])
    with col_arch1:
        st.markdown("### ⚙️ How the Pipeline Operates")
        st.markdown("""
        1. **Data Ingestion & Cleaning**: Ingestion of verified educational records from the UCI Machine Learning Repository.
        2. **Leakage Protection Protocol**: The target variable is derived strictly from final grade `G3`. Both `G3` and second-period grade `G2` are rigorously isolated from the predictor feature set to prevent trivial high-correlation data leakage.
        3. **Stratified Split**: Preserves balanced class ratios across 80% training (316 samples) and 20% test partitions (79 samples).
        4. **Preprocessed Pipeline**: Scikit-learn `ColumnTransformer` performs median imputation and standard scaling for numerical features, combined with one-hot encoding for categorical factors.
        5. **Random Forest Ensemble**: An ensemble of 100 de-correlated decision trees aggregates predictions via majority voting.
        6. **Actionable Feedback**: Translates prediction probabilities into concrete pedagogical recommendations.
        """)
    with col_arch2:
        arch_img = "assets/charts/system_architecture.png"
        if os.path.exists(arch_img):
            st.image(arch_img, caption="End-to-End System Architecture", use_container_width=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Crucial Responsible AI Disclaimer:</strong><br>
        Predictions produced by this system represent probabilistic estimates derived from historical patterns. 
        They do not constitute deterministic judgments of student intelligence, capability, or character. 
        This tool must never be used for automated administrative penalties, academic tracking, or gatekeeping. 
        It is engineered strictly as an early-warning diagnostic aid for human educators and academic counselors.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: STUDENT PREDICTION
# ==========================================
elif selected_page == "🎯 Predict Performance":
    st.title("🎯 Student Performance Prediction")
    st.markdown("Enter student academic, behavioral, and demographic features to evaluate academic risk and receive recommendations.")

    predictor = get_predictor()

    # Preset selector for fast demonstration
    st.subheader("⚡ Quick Demonstration Profiles")
    preset_choice = st.radio(
        "Load a sample profile:",
        ["Custom Manual Input", "Profile A: High Achiever", "Profile B: Moderate Passing Student", "Profile C: At-Risk Student"],
        horizontal=True
    )

    # Defaults based on preset
    if preset_choice == "Profile A: High Achiever":
        init_g1 = 16.0
        init_studytime = 3
        init_failures = 0
        init_absences = 2
        init_higher = "yes"
        init_schoolsup = "no"
        init_famsup = "yes"
        init_goout = 2
        init_medu = 4
        init_fedu = 4
    elif preset_choice == "Profile B: Moderate Passing Student":
        init_g1 = 11.0
        init_studytime = 2
        init_failures = 0
        init_absences = 6
        init_higher = "yes"
        init_schoolsup = "no"
        init_famsup = "yes"
        init_goout = 3
        init_medu = 2
        init_fedu = 2
    elif preset_choice == "Profile C: At-Risk Student":
        init_g1 = 6.0
        init_studytime = 1
        init_failures = 2
        init_absences = 16
        init_higher = "no"
        init_schoolsup = "yes"
        init_famsup = "no"
        init_goout = 5
        init_medu = 1
        init_fedu = 1
    else:
        init_g1 = 10.0
        init_studytime = 2
        init_failures = 0
        init_absences = 4
        init_higher = "yes"
        init_schoolsup = "no"
        init_famsup = "yes"
        init_goout = 3
        init_medu = 2
        init_fedu = 2

    st.markdown("---")

    with st.form("prediction_form"):
        st.subheader("1. Academic Diagnostic & Habits")
        c1, c2, c3 = st.columns(3)
        with c1:
            g1_val = st.slider("First Period Score (G1, 0–20 scale)", min_value=0.0, max_value=20.0, value=float(init_g1), step=0.5,
                               help="Diagnostic baseline grade obtained at the end of the 1st academic term.")
        with c2:
            study_map = {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}
            study_val = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4], index=init_studytime - 1,
                                     format_func=lambda x: study_map[x],
                                     help="Self-reported structured study time per week outside of class.")
        with c3:
            fail_val = st.selectbox("Past Class Failures", options=[0, 1, 2, 3, 4], index=min(init_failures, 4),
                                    help="Number of previous academic course failures (0 to 4).")

        st.subheader("2. Attendance & Educational Environment")
        c4, c5, c6 = st.columns(3)
        with c4:
            absences_val = st.number_input("School Absences (Days)", min_value=0, max_value=93, value=int(init_absences),
                                           help="Recorded number of days absent throughout the academic terms.")
        with c5:
            higher_val = st.selectbox("Aspirations for Higher Education", options=["yes", "no"],
                                      index=0 if init_higher == "yes" else 1,
                                      help="Does the student aspire to attend higher education / university?")
        with c6:
            schoolsup_val = st.selectbox("Extra Educational Support", options=["no", "yes"],
                                         index=0 if init_schoolsup == "no" else 1,
                                         help="Is the student enrolled in supplemental school support / remedial classes?")

        st.subheader("3. Socio-Demographic & Lifestyle Factors")
        c7, c8, c9, c10 = st.columns(4)
        with c7:
            medu_map = {0: "None", 1: "Primary (4th grade)", 2: "5th-9th grade", 3: "Secondary", 4: "Higher Education"}
            medu_val = st.selectbox("Mother's Education (Medu)", options=[0, 1, 2, 3, 4], index=init_medu,
                                    format_func=lambda x: medu_map[x])
        with c8:
            fedu_val = st.selectbox("Father's Education (Fedu)", options=[0, 1, 2, 3, 4], index=init_fedu,
                                    format_func=lambda x: medu_map[x])
        with c9:
            goout_val = st.slider("Going Out with Friends", min_value=1, max_value=5, value=init_goout,
                                  help="Frequency of social outings (1 = very low, 5 = very high).")
        with c10:
            health_val = st.slider("Current Health Status", min_value=1, max_value=5, value=4,
                                   help="Self-reported health condition (1 = very poor, 5 = very good).")

        # Additional background features collapsible
        with st.expander("➕ Additional Background Parameters (Demographics, Family, Alcohol Consumption)"):
            ca, cb, cc, cd = st.columns(4)
            with ca:
                sex_val = st.selectbox("Sex", options=["F", "M"], index=0)
                age_val = st.number_input("Age", min_value=15, max_value=22, value=17)
                address_val = st.selectbox("Home Address Type", options=["U", "R"], index=0, format_func=lambda x: "Urban" if x == "U" else "Rural")
            with cb:
                famsup_val = st.selectbox("Family Educational Support", options=["yes", "no"], index=0 if init_famsup == "yes" else 1)
                paid_val = st.selectbox("Extra Paid Tutoring Classes", options=["no", "yes"], index=0)
                activities_val = st.selectbox("Extracurricular Activities", options=["yes", "no"], index=0)
            with cc:
                internet_val = st.selectbox("Internet Access at Home", options=["yes", "no"], index=0)
                romantic_val = st.selectbox("Romantic Relationship", options=["no", "yes"], index=0)
                famrel_val = st.slider("Family Relationship Quality", min_value=1, max_value=5, value=4)
            with cd:
                freetime_val = st.slider("Free Time After School", min_value=1, max_value=5, value=3)
                dalc_val = st.slider("Workday Alcohol (1-5)", min_value=1, max_value=5, value=1)
                walc_val = st.slider("Weekend Alcohol (1-5)", min_value=1, max_value=5, value=1)

        submitted = st.form_submit_button("🔮 Predict Performance Tier", use_container_width=True)

    if submitted:
        input_record = {
            'age': age_val,
            'sex': sex_val,
            'address': address_val,
            'famsize': 'GT3',
            'Pstatus': 'T',
            'Medu': medu_val,
            'Fedu': fedu_val,
            'Mjob': 'services',
            'Fjob': 'other',
            'reason': 'reputation',
            'guardian': 'mother',
            'traveltime': 1,
            'studytime': study_val,
            'failures': fail_val,
            'schoolsup': schoolsup_val,
            'famsup': famsup_val,
            'paid': paid_val,
            'activities': activities_val,
            'nursery': 'yes',
            'higher': higher_val,
            'internet': internet_val,
            'romantic': romantic_val,
            'famrel': famrel_val,
            'freetime': freetime_val,
            'goout': goout_val,
            'Dalc': dalc_val,
            'Walc': walc_val,
            'health': health_val,
            'absences': absences_val,
            'G1': g1_val
        }

        res = predictor.predict(input_record)

        st.markdown("---")
        st.subheader("📋 Prediction Results & Diagnostic Insights")

        r_col1, r_col2 = st.columns([1, 1])

        with r_col1:
            tier = res['predicted_category']
            badge_class = "badge-high" if "High" in tier else ("badge-medium" if "Medium" in tier else "badge-low")
            
            st.markdown(f"**Predicted Performance Category:**")
            st.markdown(f"<div class='{badge_class}' style='font-size: 1.25rem;'>{tier}</div>", unsafe_allow_html=True)
            st.markdown(f"**Model Confidence:** `{res['confidence']:.1%}`")
            st.markdown(f"**Risk Evaluation:** `{res['risk_status']}`")

            st.markdown("#### Calibrated Class Probabilities:")
            for c_name, p_val in res['probabilities'].items():
                st.write(f"**{c_name}**: `{p_val:.1%}`")
                st.progress(float(p_val))

        with r_col2:
            st.markdown("#### 💡 Targeted Pedagogical Recommendations:")
            for rec in res['recommendations']:
                st.info(f"📌 {rec}")

        st.markdown(f"""
        <div class="disclaimer-box">
            <strong>Advisory Note:</strong> {res['disclaimer']}
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 3: DATASET EXPLORER
# ==========================================
elif selected_page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown("Inspect features, distributions, and summary statistics of the UCI Student Performance Dataset.")

    df_raw = get_raw_dataset()

    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    with e_col1:
        st.metric("Total Records", f"{df_raw.shape[0]} students")
    with e_col2:
        st.metric("Raw Attributes", f"{df_raw.shape[1]} columns")
    with e_col3:
        st.metric("Missing Values", f"{df_raw.isna().sum().sum()} nulls")
    with e_col4:
        st.metric("Mean Final Grade (G3)", f"{df_raw['G3'].mean():.2f} / 20")

    st.subheader("1. Sample Dataset Rows")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("2. Summary Descriptive Statistics (Numerical Features)")
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.subheader("3. Key Feature Distributions")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df_raw['G3'], kde=True, bins=20, color='#2563EB', ax=ax1)
        ax1.set_title("Distribution of Final Grades (G3, 0–20 Scale)")
        ax1.set_xlabel("Final Grade (G3)")
        ax1.axvline(10, color='red', linestyle='--', label='Pass/Fail Threshold (10)')
        ax1.legend()
        st.pyplot(fig1)

    with f_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='studytime', y='G3', data=df_raw, palette='Blues', ax=ax2)
        ax2.set_title("Final Grade by Weekly Study Time Category")
        ax2.set_xlabel("Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
        ax2.set_ylabel("Final Grade (G3)")
        st.pyplot(fig2)

    st.subheader("4. Target Categorization Stratification")
    dist_img = "assets/charts/class_distribution.png"
    if os.path.exists(dist_img):
        st.image(dist_img, caption="Distribution of Derived Academic Performance Tiers", use_container_width=True)


# ==========================================
# PAGE 4: MODEL EVALUATION
# ==========================================
elif selected_page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation & Empirical Results")
    st.markdown("Detailed quantitative performance analysis on unseen held-out test data (N=79 students).")

    meta = get_metrics_metadata()
    if meta:
        rf_m = meta['metrics']
        base_m = meta['baseline_metrics']

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Test Accuracy", f"{rf_m['accuracy']:.2%}", delta=f"{(rf_m['accuracy'] - base_m['accuracy']):.2%} vs Baseline")
        with m2:
            st.metric("Macro F1-Score", f"{rf_m['f1_macro']:.3f}", delta=f"{(rf_m['f1_macro'] - base_m['f1_macro']):.3f} vs Baseline")
        with m3:
            st.metric("Weighted F1-Score", f"{rf_m['f1_weighted']:.3f}")
        with m4:
            st.metric("Macro Precision", f"{rf_m['precision_macro']:.3f}")

        st.markdown("---")
        c_left, c_right = st.columns(2)

        with c_left:
            st.subheader("Confusion Matrix (Held-out Test Set)")
            cm_img = "assets/charts/confusion_matrix.png"
            if os.path.exists(cm_img):
                st.image(cm_img, use_container_width=True)
            else:
                st.write(rf_m['confusion_matrix'])

        with c_right:
            st.subheader("Top Informative Features (Gini Importance)")
            fi_img = "assets/charts/feature_importance.png"
            if os.path.exists(fi_img):
                st.image(fi_img, use_container_width=True)

        st.subheader("Detailed Classification Report")
        report_df = pd.DataFrame(rf_m['classification_report_dict']).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

        st.subheader("Comparative Benchmark: Majority Baseline vs Random Forest")
        comp_img = "assets/charts/model_comparison.png"
        if os.path.exists(comp_img):
            st.image(comp_img, use_container_width=True)

        st.markdown("""
        ### 🔍 Metric Interpretation in Plain English:
        - **Accuracy (67.1%)**: The percentage of test students correctly classified into their actual academic tier. Random Forest outperforms the trivial majority baseline by over 25 percentage points.
        - **Macro F1-Score (0.685)**: The unweighted mean of harmonic means across all three tiers. This guarantees that minority tiers (such as High Performers) are evaluated equally rather than being overshadowed by larger categories.
        - **Feature Importance**: Diagnostic test score `G1`, past course failures `failures`, and school absenteeism `absences` provide the highest predictive signal, closely mirroring educational domain literature.
        """)
    else:
        st.warning("Metrics metadata not yet loaded. Please run 'python train_model.py' to generate.")


# ==========================================
# PAGE 5: ABOUT
# ==========================================
elif selected_page == "ℹ️ About Project":
    st.title("ℹ️ About the Project")
    st.markdown("### Fundamentals of Artificial Intelligence — Course Project")

    st.markdown("""
    #### 1. Project Objectives
    - Implement a supervised machine learning algorithm to solve a real-world educational challenge.
    - Provide early identification of students at risk of underperformance to facilitate timely remedial intervention.
    - Design a rigorous, leak-free preprocessing and modeling pipeline using Scikit-Learn and Joblib.
    - Build an accessible, responsive analytics dashboard in Streamlit for academic stakeholders.

    #### 2. Algorithm Rationale: Random Forest Classifier
    - **Ensemble Bagging**: Generates an ensemble of de-correlated decision trees trained on bootstrap samples.
    - **Variance Reduction**: Aggregates predictions across 100 estimators, substantially mitigating individual tree overfitting.
    - **Feature Subsampling**: Evaluates a random subset of $\\sqrt{p}$ features at each split, ensuring robust multi-modal feature usage.
    - **Non-Linear Interactions**: Naturally captures non-linear relationships and interactions among socio-economic, attendance, and academic indicators without rigid parametric assumptions.

    #### 3. Data Leakage Prevention Strategy
    In educational data science, utilizing concurrent or subsequent examination marks (such as $G2$ or $G3$) to predict student performance constitutes severe target leakage. This system enforces strict boundary protocols:
    - **Target Derivation**: Generated exclusively from final grade $G3$.
    - **Feature Isolation**: Both $G3$ and $G2$ are excluded from the model matrix $X$.
    - **Diagnostic Window**: Prediction utilizes only early 1st-period diagnostics ($G1$) alongside historic and environmental indicators.

    #### 4. Limitations & Threats to Validity
    - **Sample Scale**: The UCI dataset contains 395 students from two Portuguese secondary schools. Findings reflect this specific demographic and curricular structure.
    - **Cultural Context**: Indicators such as parental cohabitation and alcohol consumption vary widely across global educational environments.
    - **Ethical Boundaries**: Algorithmic models can amplify historical socioeconomic biases if deployed unmonitored. Human pedagogical judgment must always remain paramount.

    #### 5. Future Enhancements
    - Expansion to longitudinal student datasets spanning multi-semester tracking.
    - Integration of Explainable AI (SHAP / LIME) for personalized student risk factor explanations.
    - Automated recommendation tracking to evaluate whether pedagogical interventions improved final grades.
    """)
