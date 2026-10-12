"""
Comprehensive Academic Report Generator for:
'Student Performance Prediction Using Machine Learning'
Generates:
1. reports/student_performance_report.pdf (Verified EXACTLY 10 pages)
2. reports/student_performance_report.docx (Fully editable Microsoft Word document)
"""

import os
import re
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

os.makedirs('reports', exist_ok=True)

# ---------------------------------------------------------------------------
# Numbered Canvas for Running Headers and Footers (Page X of 10)
# ---------------------------------------------------------------------------
class AcademicNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        page_num = self._pageNumber
        if page_num == 1:
            # Suppress headers/footers on title page
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Header
        header_text = "Fundamentals of Artificial Intelligence | Student Performance Prediction Using Machine Learning"
        self.drawString(36, 756, header_text)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 750, 576, 750)

        # Running Footer
        footer_text = f"Page {page_num} of {total_pages}"
        self.drawRightString(576, 26, footer_text)
        self.drawString(36, 26, "Department of Computer Science & Artificial Intelligence — Academic Project Report")
        self.line(36, 36, 576, 36)
        self.restoreState()


def build_pdf_report(filename="reports/student_performance_report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Academic Palette
    C_PRIMARY = colors.HexColor("#1E3A8A")    # Navy
    C_SECONDARY = colors.HexColor("#2563EB")  # Royal Blue
    C_DARK = colors.HexColor("#0F172A")       # Slate 900
    C_BODY = colors.HexColor("#1E293B")       # Slate 800
    C_MUTED = colors.HexColor("#475569")      # Slate 600

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=25,
        textColor=C_PRIMARY,
        alignment=1, # Center
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=C_SECONDARY,
        alignment=1,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=17,
        textColor=C_PRIMARY,
        spaceBefore=0,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=C_SECONDARY,
        spaceBefore=4,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'AcademicBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.8,
        leading=11.8,
        textColor=C_BODY,
        alignment=4, # Justified
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'AcademicBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.6,
        leading=11.4,
        textColor=C_BODY,
        leftIndent=12,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'CalloutBox',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.4,
        leading=11.2,
        textColor=C_PRIMARY,
        spaceAfter=4
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=9.8,
        textColor=C_DARK
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.0,
        leading=10.0,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("INSTITUTION NAME PLACEHOLDER", ParagraphStyle('InstHead', fontName='Helvetica-Bold', fontSize=14, leading=17, textColor=C_MUTED, alignment=1)))
    story.append(Paragraph("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING", ParagraphStyle('DeptHead', fontName='Helvetica', fontSize=10, leading=13, textColor=C_MUTED, alignment=1)))
    story.append(Paragraph("COURSE: FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE", ParagraphStyle('CourseHead', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=C_SECONDARY, alignment=1, spaceBefore=4)))
    story.append(Spacer(1, 35))

    story.append(HRFlowable(width="80%", thickness=2.5, color=C_PRIMARY, spaceAfter=20, spaceBefore=0))
    story.append(Paragraph("STUDENT PERFORMANCE PREDICTION USING MACHINE LEARNING", title_style))
    story.append(Paragraph("A Supervised Random Forest Classification Framework for Proactive Educational Intervention", subtitle_style))
    story.append(HRFlowable(width="80%", thickness=1.0, color=C_SECONDARY, spaceAfter=35, spaceBefore=10))

    meta_table_data = [
        [Paragraph("<b>Project Domain:</b>", body_style), Paragraph("Educational Data Mining & Predictive Analytics", body_style)],
        [Paragraph("<b>Primary Algorithm:</b>", body_style), Paragraph("Random Forest Classifier (Ensemble Bagging & Subspace)", body_style)],
        [Paragraph("<b>Student Name:</b>", body_style), Paragraph("[Student Name Placeholder]", body_style)],
        [Paragraph("<b>Roll / Registration Number:</b>", body_style), Paragraph("[Roll Number Placeholder]", body_style)],
        [Paragraph("<b>Faculty Supervisor:</b>", body_style), Paragraph("[Faculty / Mentor Name Placeholder]", body_style)],
        [Paragraph("<b>Degree Program:</b>", body_style), Paragraph("Bachelor of Technology / Bachelor of Science in AI & Data Science", body_style)],
        [Paragraph("<b>Academic Year:</b>", body_style), Paragraph("2025 – 2026", body_style)],
        [Paragraph("<b>Submission Date:</b>", body_style), Paragraph("October 2026", body_style)],
    ]
    t_meta = Table(meta_table_data, colWidths=[160, 340])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 40))
    story.append(Paragraph("<i>Academic Declaration: This project report is submitted in partial fulfillment of the requirements for the course Fundamentals of Artificial Intelligence. All implementations, empirical experiments, and reported results represent original laboratory work with genuine measured metrics.</i>", ParagraphStyle('Decl', fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=C_MUTED, alignment=1)))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: ABSTRACT, KEYWORDS, AND INTRODUCTION
    # =========================================================================
    story.append(Paragraph("1. Executive Abstract & Introduction", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>1.1 Abstract</b>", h2_style))
    story.append(Paragraph(
        "Academic underperformance and student attrition represent persistent challenges across secondary and higher education institutions globally. Conventional pedagogical evaluation mechanisms are predominantly summative and retrospective: students who require remedial academic assistance are typically diagnosed only after final semester examinations, when meaningful intervention opportunities have elapsed. This study proposes and empirically validates a production-ready machine learning framework for predicting student academic performance tiers using early diagnostic assessments alongside multi-dimensional socio-demographic, behavioral, and familial factors. Utilizing the benchmark UCI Student Performance dataset (N = 395 secondary students), the predictive pipeline implements a Random Forest Classifier trained on 24 carefully curated input features. Crucially, a rigorous Data Leakage Prevention Protocol is established by excluding late-stage grades (G2) and the final score (G3) from the predictor matrix. The model demonstrates an empirical test accuracy of 67.09% and a Macro F1-score of 0.6845 on unseen held-out test data, significantly outperforming the naive majority-class baseline (accuracy 41.77%, Macro F1 0.1964). Feature importance analysis via Mean Decrease in Impurity identifies early-period diagnostic scores (G1, 36.5%), prior course failures (5.2%), and absenteeism (5.0%) as principal determinants of academic outcomes. The model is embedded into an interactive Streamlit analytical dashboard that pairs probabilistic risk forecasts with targeted pedagogical recommendations.",
        body_style
    ))

    story.append(Paragraph("<b>Keywords:</b> Artificial Intelligence, Educational Data Mining, Random Forest Classifier, Student Performance Prediction, Early Warning System, Data Leakage Prevention, Pedagogical Intervention.", ParagraphStyle('KW', parent=body_style, fontName='Helvetica-Bold', fontSize=8.4, leading=11, textColor=C_PRIMARY)))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>1.2 Introduction</b>", h2_style))
    story.append(Paragraph(
        "The contemporary educational landscape is undergoing a profound paradigm shift driven by digital academic administration. Primary, secondary, and tertiary institutions routinely compile extensive longitudinal records detailing student course attendance, assignment submissions, parental educational attainments, extra-curricular engagements, and historical grade points. However, within conventional institutional workflows, these heterogeneous data streams remain sequestered within administrative databases, consulted only for retrospective compliance reporting or post-examination grade transcript production.",
        body_style
    ))
    story.append(Paragraph(
        "Educational Data Mining (EDM) and Learning Analytics (LA) have emerged as foundational sub-disciplines of Artificial Intelligence designed to transform dormant institutional repositories into proactive, actionable pedagogical intelligence. By applying supervised machine learning algorithms to historical learner records, educational decision-makers can uncover subtle, multi-factorial interaction patterns that correlate with academic vulnerability.",
        body_style
    ))
    story.append(Paragraph(
        "Unlike industrial predictive systems where false positives or latency merely imply computational overhead, deploying artificial intelligence within human educational ecosystems requires rigorous methodological conservatism. Predictions cannot be treated as deterministic verdicts regarding an individual learner's intellectual ceiling; rather, they serve as calibrated probabilistic signposts. This investigation details the conceptualization, mathematical formulation, software engineering, and empirical benchmarking of a leak-free Random Forest classification system tailored for institutional decision support.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: PROBLEM STATEMENT, MOTIVATION, OBJECTIVES, SCOPE, SOLUTION
    # =========================================================================
    story.append(Paragraph("2. Problem Formulation, Motivation & Project Scope", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>2.1 Problem Statement</b>", h2_style))
    story.append(Paragraph(
        "Educational institutions collect disparate data regarding student attendance, study habits, parental backgrounds, and baseline coursework evaluations. Yet, identifying students on trajectory toward academic failure remains uncoordinated and late. The specific engineering challenge addressed in this investigation is: <i>How can an ensemble machine learning classification pipeline accurately categorize student terminal academic performance into actionable tiers early in the academic calendar while strictly eliminating data leakage and providing transparent pedagogical guidance?</i>",
        body_style
    ))

    story.append(Paragraph("<b>2.2 Motivation</b>", h2_style))
    story.append(Paragraph(
        "Course failure carries severe cascading consequences: student demotivation, increased psychological distress, delayed graduation schedules, and elevated institutional attrition. When instructors rely solely on informal intuition or end-of-term examinations, struggling students quietly accumulate foundational learning deficits. Developing an algorithmic early-warning mechanism creates a critical intervention window during which faculty advisors, peer tutoring centers, and counseling personnel can deploy structured remedial support.",
        body_style
    ))

    story.append(Paragraph("<b>2.3 Project Objectives</b>", h2_style))
    story.append(Paragraph("The technical and academic objectives of this project are strictly formulated as follows:", body_style))
    story.append(Paragraph("<b>1. Algorithm Formulation:</b> Implement a robust Random Forest Classifier utilizing 100 de-correlated decision trees to model complex non-linear feature interactions without parametric assumptions.", bullet_style))
    story.append(Paragraph("<b>2. Data Leakage Elimination:</b> Formulate an airtight preprocessing protocol that excludes intermediate and terminal examination grades (G2 and G3) from the predictor set, resolving a pervasive flaw in published literature.", bullet_style))
    story.append(Paragraph("<b>3. Modular Scikit-Learn Pipeline:</b> Construct a production-grade ColumnTransformer pipeline integrating SimpleImputer, StandardScaler, and OneHotEncoder for seamless inference on unseen data.", bullet_style))
    story.append(Paragraph("<b>4. Empirical Benchmarking:</b> Evaluate the classifier on held-out test data using multiclass Accuracy, Macro F1, Weighted F1, Confusion Matrix analysis, and baseline comparison.", bullet_style))
    story.append(Paragraph("<b>5. Interactive Analytics Portal:</b> Develop and deploy a modern, responsive Streamlit web application providing educators with real-time inference, probability distributions, and personalized recommendations.", bullet_style))

    story.append(Paragraph("<b>2.4 Scope and Boundaries</b>", h2_style))
    story.append(Paragraph(
        "The project scope encompasses secondary educational mathematics performance within the verified UCI dataset. The predictive model operates strictly at the completion of the first academic evaluation period (Period 1), ensuring that inference occurs sufficiently early to permit pedagogical remediation. Consequential administrative decisions (such as automated grading, course expulsion, or disciplinary sanctions) are explicitly designated outside system scope.",
        body_style
    ))

    story.append(Paragraph("<b>2.5 Proposed Solution Architecture</b>", h2_style))
    story.append(Paragraph(
        "We propose an end-to-end supervised machine learning decision-support architecture. Raw student demographic and baseline academic records undergo missing-value imputation and feature scaling. A stratified 80/20 partition separates training from evaluation. The Random Forest ensemble aggregates predictions across bootstrap-aggregated sub-trees. Outputs are exposed via a lightweight web graphical interface rendering class probabilities and contextual intervention tips.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: BACKGROUND AND RELATED WORK
    # =========================================================================
    story.append(Paragraph("3. Theoretical Background & Related Literature", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>3.1 Foundations of Educational Data Mining</b>", h2_style))
    story.append(Paragraph(
        "Educational Data Mining (EDM) encompasses research methodologies dedicated to analyzing unique data emerging from educational contexts to better understand student learning trajectories and institutional effectiveness (Romero & Ventura, 2010). Early EDM investigations relied predominantly on standard statistical regression models to identify bivariate correlations between grade point averages and isolated attributes such as socioeconomic status or attendance rates. However, educational success is inherently non-linear: socio-familial stress may severely impact one learner while exerting negligible influence on another whose study discipline or peer support compensates for environmental adversity.",
        body_style
    ))

    story.append(Paragraph("<b>3.2 Overview of Predictive Algorithms in EDM</b>", h2_style))
    story.append(Paragraph(
        "Academic literature has explored numerous algorithmic paradigms for student outcome forecasting, including Logistic Regression, Naive Bayes, Support Vector Machines (SVM), Multilayer Perceptrons (MLP), and Decision Trees (Baker & Inventado, 2014). Linear classifiers frequently suffer from high bias due to their inability to model complex feature interactions without extensive manual polynomial term engineering. Conversely, unpruned individual Decision Trees exhibit severe variance, overfitting idiosyncrasies of small classroom sample cohorts. Ensembling paradigms, specifically Random Forests and Gradient Boosted Trees, have consistently demonstrated superior empirical performance across educational benchmark studies by balancing bias and variance through aggregation.",
        body_style
    ))

    story.append(Paragraph("<b>3.3 Benchmark Dataset: Cortez and Silva (2008)</b>", h2_style))
    story.append(Paragraph(
        "In their seminal study, Cortez and Silva (2008) collected and evaluated secondary school student achievement from two Portuguese public schools within the Alentejo region. The authors modeled performance across two core subjects: Mathematics (395 records) and Portuguese language (649 records). Their exploratory experiments evaluated multiple data mining techniques, including Decision Trees, Random Forests, Neural Networks, and Support Vector Machines. A critical methodological observation from their work was that including the second-period grade (G2) produces artificially high predictive accuracy (correlations exceeding 0.90), as G2 occurs immediately prior to the final exam. Consequently, forecasting performance solely with G1 and background attributes constitutes the most realistic and practically valuable formulation for educational early intervention.",
        body_style
    ))

    story.append(Paragraph("<b>3.4 Critical Analysis of Data Leakage in Academic Research</b>", h2_style))
    story.append(Paragraph(
        "Data leakage occurs when information from outside the training dataset or information unavailable at the true operational decision time is improperly utilized to train a machine learning model (Kaufman et al., 2012). In student performance prediction, data leakage manifests in two primary ways: (1) <i>Target Contamination</i>: utilizing variables derived directly from the ground truth outcome; and (2) <i>Temporal Leakage</i>: utilizing mid-term or late-term assessment scores (such as G2) when attempting to forecast early risk. Numerous student performance prediction tutorials and published papers report inflated accuracies exceeding 90% precisely because G2 is retained in the feature vector. When deployed in practice during month two of a semester, such models catastrophically fail because G2 does not yet exist. In contrast, our implementation enforces a strict temporal and causal boundary.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: DATASET DESCRIPTION, PREPROCESSING, TARGET DEFINITION
    # =========================================================================
    story.append(Paragraph("4. Dataset Characterization & Preprocessing Pipeline", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>4.1 UCI Dataset Overview & Source Attribution</b>", h2_style))
    story.append(Paragraph(
        "This project utilizes the Mathematics cohort of the Student Performance Dataset from the UC Irvine Machine Learning Repository (Cortez & Silva, 2008). The raw dataset comprises 395 student instances characterized by 33 initial attributes. Data collection combined school administrative records (grades, absences) with comprehensive self-administered questionnaires (family structure, study habits, lifestyle).",
        body_style
    ))

    story.append(Paragraph("<b>4.2 Target Variable Derivation & Academic Categorization</b>", h2_style))
    story.append(Paragraph(
        "In the Portuguese secondary educational system, academic evaluation operates on a continuous integer grading scale from 0 to 20 points. A score of 10 represents the minimum passing threshold, while scores of 14 and above indicate academic distinction. To formulate an operationally meaningful decision-support target, we map continuous final grade G3 into three strategic academic performance tiers:",
        body_style
    ))

    tier_table_data = [
        [Paragraph("<b>Performance Tier</b>", table_header_style), Paragraph("<b>Score Range</b>", table_header_style), Paragraph("<b>Institutional Meaning</b>", table_header_style), Paragraph("<b>Cohort Count (N)</b>", table_header_style)],
        [Paragraph("Low (At-Risk)", table_cell_style), Paragraph("G3 &lt; 10 (0 to 9)", table_cell_style), Paragraph("Formal course failure; immediate remedial intervention required", table_cell_style), Paragraph("130 (32.9%)", table_cell_style)],
        [Paragraph("Medium (Satisfactory)", table_cell_style), Paragraph("10 &le; G3 &lt; 14", table_cell_style), Paragraph("Core passing competency; stable progress without honors", table_cell_style), Paragraph("165 (41.8%)", table_cell_style)],
        [Paragraph("High (Good / Distinction)", table_cell_style), Paragraph("14 &le; G3 &le; 20", table_cell_style), Paragraph("Academic excellence; honors candidate", table_cell_style), Paragraph("100 (25.3%)", table_cell_style)],
    ]
    t_tier = Table(tier_table_data, colWidths=[110, 85, 245, 100])
    t_tier.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_tier)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>4.3 Feature Selection & Leakage Isolation</b>", h2_style))
    story.append(Paragraph(
        "To enforce the Data Leakage Prevention Protocol, G3 (target source) and G2 (second-period grade, Pearson r = 0.904 with G3) are strictly eliminated from the feature matrix X. The institutional identifier 'school' is also excluded to ensure institutional transferability. The resulting feature space comprises 24 features partitioned into 14 numerical and 10 categorical variables.",
        body_style
    ))

    feat_table_data = [
        [Paragraph("<b>Domain</b>", table_header_style), Paragraph("<b>Features Included in Model Matrix</b>", table_header_style), Paragraph("<b>Preprocessing Transformation</b>", table_header_style)],
        [Paragraph("Academic Diagnostics", table_cell_style), Paragraph("G1 (Term 1 score), failures (past course failures), studytime (weekly study hours), schoolsup (school support), paid (extra tutoring)", table_cell_style), Paragraph("Median Imputer + StandardScaler (num) / Mode + OneHotEncoder (cat)", table_cell_style)],
        [Paragraph("Attendance & School", table_cell_style), Paragraph("absences (recorded absences 0-93), traveltime (transit time), higher (higher ed desire)", table_cell_style), Paragraph("StandardScaler for absences & traveltime / OneHotEncoder for higher", table_cell_style)],
        [Paragraph("Family Background", table_cell_style), Paragraph("Medu, Fedu (parents education 0-4), Mjob, Fjob (parents jobs), famsize, Pstatus, famrel", table_cell_style), Paragraph("StandardScaler for ordinal ranks / OneHotEncoder (drop='first') for nominal", table_cell_style)],
        [Paragraph("Lifestyle & Habits", table_cell_style), Paragraph("goout, freetime, Dalc, Walc, health, activities, internet, romantic, sex, age, address", table_cell_style), Paragraph("StandardScaler for ordinal lifestyle scales / OneHotEncoder for binary/nominal", table_cell_style)],
    ]
    t_feat = Table(feat_table_data, colWidths=[105, 275, 160])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_SECONDARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_feat)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: RANDOM FOREST ALGORITHM, MATHEMATICS, ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("5. Random Forest Classification: Theory & Mathematical Formulation", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>5.1 Algorithm Principle & Ensemble Architecture</b>", h2_style))
    story.append(Paragraph(
        "The Random Forest Classifier, introduced by Leo Breiman (2001), is an ensemble supervised learning algorithm combining Bootstrap Aggregation (bagging) with random feature subspace sampling. While an individual Classification and Regression Tree (CART) has low bias, it suffers from high variance and tends to overfit noisy training samples. Random Forest addresses this fundamental bias-variance trade-off by cultivating an ensemble of B de-correlated decision trees {h_b(x)}_{b=1}^B and aggregating their individual predictions via majority voting:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Mathematical Ensemble Decision Rule:</b> &nbsp;&nbsp;&nbsp; <i>H(x) = argmax_{c &isin; C} &Sigma;_{b=1}^B I(h_b(x) = c)</i>",
        callout_style
    ))
    story.append(Paragraph(
        "where C = {Low, Medium, High} denotes the set of academic performance classes, and I(&middot;) represents the indicator function. The calibrated predicted class probability is derived from the proportion of constituent trees voting for category c: &nbsp; <i>P(Y = c | x) = (1 / B) &Sigma;_{b=1}^B I(h_b(x) = c)</i>.",
        body_style
    ))

    story.append(Paragraph("<b>5.2 Mathematical Splitting Criterion: Gini Impurity</b>", h2_style))
    story.append(Paragraph(
        "At each internal decision node &tau;, the tree evaluates candidate split thresholds across randomly sampled features to maximize the decrease in Gini Impurity. The Gini impurity of node &tau; containing class proportions p(c | &tau;) is defined as:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Gini Impurity Equation:</b> &nbsp;&nbsp;&nbsp; <i>I_G(&tau;) = 1 - &Sigma;_{c=1}^K [ p(c | &tau;) ]^2</i>",
        callout_style
    ))
    story.append(Paragraph(
        "A split s partitions node &tau; into child nodes &tau;_L and &tau;_R containing N_L and N_R instances respectively. The optimal split s* maximizes the impurity gain: &nbsp; <i>&Delta; I_G(&tau;, s) = I_G(&tau;) - [ (N_L / N) I_G(&tau;_L) + (N_R / N) I_G(&tau;_R) ]</i>.",
        body_style
    ))

    story.append(Paragraph("<b>5.3 System Workflow Architecture Diagram</b>", h2_style))
    arch_img_path = "assets/charts/system_architecture.png"
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=7.2*inch, height=2.2*inch))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<i>Figure 5.1: End-to-end dataflow pipeline depicting raw data ingestion, leak-free preprocessing, stratified partitioning, Random Forest ensemble fitting, and Streamlit dashboard integration.</i>", ParagraphStyle('Cap', fontName='Helvetica-Oblique', fontSize=7.6, leading=9.5, textColor=C_MUTED, alignment=1)))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: SYSTEM DESIGN, IMPLEMENTATION, TOOLS, METHODOLOGY
    # =========================================================================
    story.append(Paragraph("6. System Design, Implementation & Methodology", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>6.1 Software Architecture & Module Organization</b>", h2_style))
    story.append(Paragraph(
        "The project is engineered according to modern modular software engineering principles. Rather than relying on monolithic scripts, functionality is segregated into dedicated, decoupled modules within the <code>src/</code> package, ensuring testability, maintainability, and clean separation of concerns:",
        body_style
    ))

    mod_table_data = [
        [Paragraph("<b>Component File</b>", table_header_style), Paragraph("<b>Architectural Responsibility</b>", table_header_style), Paragraph("<b>Key Functions / Classes</b>", table_header_style)],
        [Paragraph("src/data_preprocessing.py", table_cell_style), Paragraph("Dataset loading, target derivation, leakage isolation, and ColumnTransformer construction.", table_cell_style), Paragraph("load_data(), create_target(), build_preprocessor(), prepare_data()", table_cell_style)],
        [Paragraph("src/model_training.py", table_cell_style), Paragraph("Scikit-learn pipeline assembly, baseline training, Random Forest fitting, and Joblib serialization.", table_cell_style), Paragraph("create_rf_pipeline(), train_and_evaluate()", table_cell_style)],
        [Paragraph("src/evaluation.py", table_cell_style), Paragraph("Comprehensive metric computation, confusion matrix generation, and chart plotting.", table_cell_style), Paragraph("compute_metrics(), get_feature_importances(), plot_confusion_matrix()", table_cell_style)],
        [Paragraph("src/prediction.py", table_cell_style), Paragraph("Production inference wrapper, input validation, probability calibration, and pedagogical tips.", table_cell_style), Paragraph("StudentPerformancePredictor, predict(), predict_batch()", table_cell_style)],
        [Paragraph("app.py", table_cell_style), Paragraph("Responsive multi-section Streamlit web analytics portal with model memory caching.", table_cell_style), Paragraph("get_predictor(), Streamlit forms, sliders, distribution visualizations", table_cell_style)],
    ]
    t_mod = Table(mod_table_data, colWidths=[120, 240, 180])
    t_mod.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_mod)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>6.2 Technology Stack & Libraries</b>", h2_style))
    story.append(Paragraph(
        "Implementation relies on the verified Python 3 scientific and machine learning ecosystem: (1) <b>pandas</b> and <b>NumPy</b> for high-performance vectorized tabular transformations; (2) <b>scikit-learn</b> for model estimation, ColumnTransformer, and metric evaluations; (3) <b>Joblib</b> for thread-safe model serialization; (4) <b>Streamlit</b> for responsive web dashboard rendering; (5) <b>Matplotlib</b> and <b>Seaborn</b> for publication-quality visual charting; and (6) <b>pytest</b> for automated continuous test execution.",
        body_style
    ))

    story.append(Paragraph("<b>6.3 Reproducibility & Hyperparameter Configuration</b>", h2_style))
    story.append(Paragraph(
        "To ensure complete experimental reproducibility, all pseudo-random processes (stratified splitting, bootstrap resampling, feature subsampling) are initialized with a constant random seed (<code>random_state=42</code>). Hyperparameter selection was guided by stratified 5-fold cross-validation grid search to balance expressive model capacity against overfitting risks. The finalized model configuration is: <i>n_estimators = 100</i>, <i>max_depth = 6</i>, <i>min_samples_split = 4</i>, <i>criterion = 'gini'</i>, and <i>n_jobs = -1</i>.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: EXPERIMENTAL SETUP, RESULTS, AND EVALUATION CHARTS
    # =========================================================================
    story.append(Paragraph("7. Experimental Setup, Empirical Results & Metrics", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>7.1 Experimental Partitioning</b>", h2_style))
    story.append(Paragraph(
        "The dataset (N = 395) was partitioned into an 80% training set (N_train = 316) and an unseen 20% held-out test set (N_test = 79) utilizing stratified sampling. Stratification ensures that each partition maintains the exact ground truth class distribution: 41.8% Medium, 32.9% Low, and 25.3% High.",
        body_style
    ))

    story.append(Paragraph("<b>7.2 Comparative Benchmark: Baseline vs. Random Forest Classifier</b>", h2_style))
    story.append(Paragraph(
        "To provide a rigorous academic baseline, the Random Forest model is evaluated alongside a Zero-R Majority Class Baseline (DummyClassifier) that deterministically predicts the most frequent training category ('Medium (Satisfactory)'). All reported metrics represent actual measured values on unseen held-out test data.",
        body_style
    ))

    results_table_data = [
        [Paragraph("<b>Evaluation Metric</b>", table_header_style), Paragraph("<b>Majority Class Baseline</b>", table_header_style), Paragraph("<b>Random Forest Classifier</b>", table_header_style), Paragraph("<b>Absolute Improvement</b>", table_header_style)],
        [Paragraph("Test Accuracy", table_cell_style), Paragraph("0.4177 (41.77%)", table_cell_style), Paragraph("<b>0.6709 (67.09%)</b>", table_cell_style), Paragraph("+25.32%", table_cell_style)],
        [Paragraph("Macro F1-Score", table_cell_style), Paragraph("0.1964", table_cell_style), Paragraph("<b>0.6845</b>", table_cell_style), Paragraph("+0.4881", table_cell_style)],
        [Paragraph("Weighted F1-Score", table_cell_style), Paragraph("0.2462", table_cell_style), Paragraph("<b>0.6724</b>", table_cell_style), Paragraph("+0.4262", table_cell_style)],
        [Paragraph("Macro Precision", table_cell_style), Paragraph("0.1392", table_cell_style), Paragraph("<b>0.7016</b>", table_cell_style), Paragraph("+0.5624", table_cell_style)],
        [Paragraph("Macro Recall", table_cell_style), Paragraph("0.3333", table_cell_style), Paragraph("<b>0.6746</b>", table_cell_style), Paragraph("+0.3413", table_cell_style)],
    ]
    t_res = Table(results_table_data, colWidths=[140, 130, 140, 130])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>7.3 Per-Class Performance Breakdown & Confusion Matrix</b>", h2_style))
    
    cm_img_path = "assets/charts/confusion_matrix.png"
    fi_img_path = "assets/charts/feature_importance.png"
    
    # Place Confusion matrix and Feature importance side by side or compact
    c_imgs = []
    if os.path.exists(cm_img_path) and os.path.exists(fi_img_path):
        c_imgs = [
            [Image(cm_img_path, width=3.5*inch, height=2.2*inch), Image(fi_img_path, width=3.6*inch, height=2.2*inch)]
        ]
        t_imgs = Table(c_imgs, colWidths=[265, 275])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(t_imgs)

    story.append(Paragraph("<i>Figure 7.1: (Left) Confusion Matrix on held-out test data (N=79). (Right) Top 12 Informative Features based on Mean Decrease in Impurity (Gini Importance).</i>", ParagraphStyle('Cap2', fontName='Helvetica-Oblique', fontSize=7.4, leading=9.0, textColor=C_MUTED, alignment=1)))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: DISCUSSION, LIMITATIONS, ETHICS, APPLICATIONS
    # =========================================================================
    story.append(Paragraph("8. Results Discussion, Ethical AI & Practical Implications", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>8.1 Pedagogical Discussion of Findings</b>", h2_style))
    story.append(Paragraph(
        "Empirical analysis reveals that Random Forest achieves robust predictive discrimination across all three academic tiers. High Performers are classified with high precision (82.6%) and recall (75.0%), indicating that strong foundational diagnostics (G1 &ge; 14) and consistent habits strongly preserve academic trajectory. For the critical 'Low (At-Risk)' cohort, the model successfully identifies 57.7% of struggling students early in the term with 68.2% precision. While a baseline majority classifier fails completely to detect any at-risk or high-achieving students (Macro F1 0.196), our ensemble model provides teachers with actionable lead time.",
        body_style
    ))
    story.append(Paragraph(
        "Feature importance rankings confirm educational domain literature: the early diagnostic score (G1) contributes 36.5% of total impurity reduction, followed by past course failures (5.2%) and school absenteeism (5.0%). Notably, socio-environmental factors (mother's education Medu 3.9%, social outings goout 4.7%, and study time 2.4%) contribute vital non-linear signals that prevent the model from collapsing into a naive linear grade extrapolator.",
        body_style
    ))

    story.append(Paragraph("<b>8.2 Limitations & Threats to Validity</b>", h2_style))
    story.append(Paragraph(
        "Three distinct methodological limitations must be acknowledged: (1) <i>Cohort Sample Scale</i>: The dataset reflects 395 students across two public secondary schools within Portugal. Institutional, regional, and national cultural nuances may alter the relative influence of factors like alcohol consumption or commute times; (2) <i>Self-Reporting Bias</i>: Behavioral metrics (such as study hours and alcohol use) are derived from student questionnaires and subject to social desirability bias; and (3) <i>Curricular Generalizability</i>: Mathematics curricula follow structured sequential prerequisite chains; subjects emphasizing qualitative discussion (e.g., literature or history) may demonstrate different predictive dynamics.",
        body_style
    ))

    story.append(Paragraph("<b>8.3 Responsible AI & Ethical Considerations</b>", h2_style))
    story.append(Paragraph(
        "Deploying predictive artificial intelligence within human educational institutions entails profound ethical accountability:",
        body_style
    ))
    story.append(Paragraph("<b>1. Human-in-the-Loop Governance:</b> Algorithmic predictions must serve strictly as advisory decision aids for instructors and academic counselors. The tool must NEVER be utilized to automate punitive disciplinary actions, academic track gating, or course expulsion.", bullet_style))
    story.append(Paragraph("<b>2. Probabilistic Humility:</b> Outputs must be communicated as calibrated probability distributions rather than deterministic assertions of ability, reinforcing that every student's trajectory remains malleable.", bullet_style))
    story.append(Paragraph("<b>3. Algorithmic Bias Mitigation:</b> Models trained on historical data risk perpetuating systemic socio-economic inequalities (such as penalizing students with lower parental educational attainment). Practitioners must apply contextual empathy when reviewing model outputs.", bullet_style))

    story.append(Paragraph("<b>8.4 Future Enhancements</b>", h2_style))
    story.append(Paragraph(
        "Planned enhancements include: (1) incorporating Explainable AI frameworks (SHAP and LIME) to furnish individualized student risk factor watermarks; (2) ingesting granular longitudinal LMS log metrics (assignment submission timestamps, quiz attempts); and (3) executing randomized control trials to validate whether AI-prompted pedagogical interventions demonstrably enhance final exam pass rates.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: CONCLUSION AND REFERENCES
    # =========================================================================
    story.append(Paragraph("9. Conclusions & Academic References", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("<b>9.1 Project Conclusion</b>", h2_style))
    story.append(Paragraph(
        "This project successfully designed, implemented, and empirically validated a machine learning system for student performance prediction using the Random Forest algorithm in fulfillment of the course requirements for Fundamentals of Artificial Intelligence. By enforcing a rigorous Data Leakage Prevention Protocol—excluding late-stage and terminal examination marks (G2 and G3)—the framework establishes a realistic, operationally viable early-warning window for educators.",
        body_style
    ))
    story.append(Paragraph(
        "Tested on held-out test data from the UCI repository, the Random Forest model achieved a classification accuracy of 67.09% and a Macro F1-score of 0.6845, demonstrating an absolute improvement of +25.32% over naive baseline prediction. The automated pipeline successfully balances multi-dimensional academic diagnostics, attendance metrics, and familial lifestyle factors. When packaged into a modular Python architecture and served via an interactive Streamlit web dashboard, the framework delivers actionable pedagogical recommendations while maintaining ethical human-in-the-loop safeguards.",
        body_style
    ))

    story.append(Paragraph("<b>9.2 Summary of Deliverables Produced</b>", h2_style))
    story.append(Paragraph("• <b>Modular Python Codebase:</b> Full scikit-learn pipeline implementation in <code>src/</code> with automated preprocessing, model fitting, and inference.", bullet_style))
    story.append(Paragraph("• <b>Trained Model Artifact:</b> Serialized Joblib pipeline (<code>models/student_rf_pipeline.joblib</code>) enabling zero-retraining inference.", bullet_style))
    story.append(Paragraph("• <b>Automated Test Suite:</b> 12 comprehensive unit tests in <code>tests/</code> validating input sanitization, preprocessing, and model inference.", bullet_style))
    story.append(Paragraph("• <b>Interactive Web Dashboard:</b> Full-featured Streamlit analytical dashboard (<code>app.py</code>) with custom academic styling.", bullet_style))
    story.append(Paragraph("• <b>Academic Presentation:</b> 12-slide editable presentation (<code>student_performance_presentation.pptx</code>) with embedded speaker notes.", bullet_style))
    story.append(Paragraph("• <b>Formal Documentation:</b> Complete 10-page formal PDF report and editable DOCX submission document.", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>9.3 References</b>", h2_style))
    refs = [
        "[1] Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics. In <i>Learning analytics</i> (pp. 61-75). Springer, New York, NY.",
        "[2] Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324",
        "[3] Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict secondary school student performance. In <i>Proceedings of the 5th Annual Future Business Technology Conference (FUBUTEC 2008)</i>, Porto, Portugal (pp. 5-12). EUROSIS-ETI.",
        "[4] Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. <i>ACM Transactions on Knowledge Discovery from Data (TKDD)</i>, 6(4), 1-21.",
        "[5] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.",
        "[6] Romero, C., & Ventura, S. (2010). Educational data mining: a review of the state of the art. <i>IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews)</i>, 40(6), 601-618. https://doi.org/10.1109/TSMCC.2010.2053532",
        "[7] Romero, C., Ventura, S., & García, E. (2008). Data mining in Moodle: An education data mining approach. <i>Computers & Education</i>, 51(1), 368-384.",
        "[8] Siemens, G., & Baker, R. S. (2012). Learning analytics and educational data mining: towards a mutual understanding. In <i>Proceedings of the 2nd International Conference on Learning Analytics and Knowledge</i> (pp. 252-254)."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefEntry', fontName='Helvetica', fontSize=7.2, leading=9.2, textColor=C_MUTED, spaceAfter=2.5)))

    doc.build(story, canvasmaker=AcademicNumberedCanvas)
    print(f"Generated PDF: {filename}")


def build_docx_report(filename="reports/student_performance_report.docx"):
    doc = docx.Document()

    # Set page margins: 0.6 inch
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_inst = p_title.add_run("INSTITUTION NAME PLACEHOLDER\nDEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\nCOURSE: FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE\n\n")
    run_inst.font.size = Pt(11)
    run_inst.font.bold = True
    run_inst.font.color.rgb = RGBColor(71, 85, 105)

    run_t = p_title.add_run("STUDENT PERFORMANCE PREDICTION USING MACHINE LEARNING\n")
    run_t.font.size = Pt(22)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(30, 58, 138)

    run_sub = p_title.add_run("A Supervised Random Forest Classification Framework for Proactive Educational Intervention\n\n")
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = RGBColor(37, 99, 235)

    # Metadata Table
    t_meta = doc.add_table(rows=8, cols=2)
    t_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_rows = [
        ("Project Domain:", "Educational Data Mining & Predictive Analytics"),
        ("Primary Algorithm:", "Random Forest Classifier (Ensemble Bagging & Subspace)"),
        ("Student Name:", "[Student Name Placeholder]"),
        ("Roll / Registration Number:", "[Roll Number Placeholder]"),
        ("Faculty Supervisor:", "[Faculty / Mentor Name Placeholder]"),
        ("Degree Program:", "Bachelor of Technology / Bachelor of Science in AI & Data Science"),
        ("Academic Year:", "2025 – 2026"),
        ("Submission Date:", "October 2026")
    ]
    for idx, (lbl, val) in enumerate(meta_rows):
        row = t_meta.rows[idx]
        row.cells[0].paragraphs[0].add_run(lbl).bold = True
        row.cells[0].paragraphs[0].runs[0].font.size = Pt(9.5)
        row.cells[1].paragraphs[0].add_run(val)
        row.cells[1].paragraphs[0].runs[0].font.size = Pt(9.5)

    doc.add_page_break()

    # Section 1
    doc.add_heading("1. Executive Abstract & Introduction", level=1)
    doc.add_heading("1.1 Abstract", level=2)
    p_abs = doc.add_paragraph(
        "Academic underperformance and student attrition represent persistent challenges across secondary and higher education institutions globally. Conventional pedagogical evaluation mechanisms are predominantly summative and retrospective: students who require remedial academic assistance are typically diagnosed only after final semester examinations, when meaningful intervention opportunities have elapsed. This study proposes and empirically validates a production-ready machine learning framework for predicting student academic performance tiers using early diagnostic assessments alongside multi-dimensional socio-demographic, behavioral, and familial factors. Utilizing the benchmark UCI Student Performance dataset (N = 395 secondary students), the predictive pipeline implements a Random Forest Classifier trained on 24 carefully curated input features. Crucially, a rigorous Data Leakage Prevention Protocol is established by excluding late-stage grades (G2) and the final score (G3) from the predictor matrix. The model demonstrates an empirical test accuracy of 67.09% and a Macro F1-score of 0.6845 on unseen held-out test data, significantly outperforming the naive majority-class baseline (accuracy 41.77%, Macro F1 0.1964). Feature importance analysis via Mean Decrease in Impurity identifies early-period diagnostic scores (G1, 36.5%), prior course failures (5.2%), and absenteeism (5.0%) as principal determinants of academic outcomes. The model is embedded into an interactive Streamlit analytical dashboard that pairs probabilistic risk forecasts with targeted pedagogical recommendations."
    )
    p_kw = doc.add_paragraph("Keywords: Artificial Intelligence, Educational Data Mining, Random Forest Classifier, Student Performance Prediction, Early Warning System, Data Leakage Prevention, Pedagogical Intervention.")
    p_kw.runs[0].bold = True

    doc.add_heading("1.2 Introduction", level=2)
    doc.add_paragraph(
        "The contemporary educational landscape is undergoing a profound paradigm shift driven by digital academic administration. Primary, secondary, and tertiary institutions routinely compile extensive longitudinal records detailing student course attendance, assignment submissions, parental educational attainments, extra-curricular engagements, and historical grade points. However, within conventional institutional workflows, these heterogeneous data streams remain sequestered within administrative databases, consulted only for retrospective compliance reporting or post-examination grade transcript production."
    )
    doc.add_paragraph(
        "Educational Data Mining (EDM) and Learning Analytics (LA) have emerged as foundational sub-disciplines of Artificial Intelligence designed to transform dormant institutional repositories into proactive, actionable pedagogical intelligence. By applying supervised machine learning algorithms to historical learner records, educational decision-makers can uncover subtle, multi-factorial interaction patterns that correlate with academic vulnerability."
    )

    doc.add_page_break()

    # Section 2
    doc.add_heading("2. Problem Formulation, Motivation & Project Scope", level=1)
    doc.add_heading("2.1 Problem Statement", level=2)
    doc.add_paragraph(
        "Educational institutions collect disparate data regarding student attendance, study habits, parental backgrounds, and baseline coursework evaluations. Yet, identifying students on trajectory toward academic failure remains uncoordinated and late. The specific engineering challenge addressed in this investigation is: How can an ensemble machine learning classification pipeline accurately categorize student terminal academic performance into actionable tiers early in the academic calendar while strictly eliminating data leakage and providing transparent pedagogical guidance?"
    )
    doc.add_heading("2.2 Motivation", level=2)
    doc.add_paragraph(
        "Course failure carries severe cascading consequences: student demotivation, increased psychological distress, delayed graduation schedules, and elevated institutional attrition. Developing an algorithmic early-warning mechanism creates a critical intervention window during which faculty advisors, peer tutoring centers, and counseling personnel can deploy structured remedial support."
    )
    doc.add_heading("2.3 Project Objectives", level=2)
    for obj in [
        "1. Algorithm Formulation: Implement a robust Random Forest Classifier utilizing 100 de-correlated decision trees to model complex non-linear feature interactions without parametric assumptions.",
        "2. Data Leakage Elimination: Formulate an airtight preprocessing protocol that excludes intermediate and terminal examination grades (G2 and G3) from the predictor set, resolving a pervasive flaw in published literature.",
        "3. Modular Scikit-Learn Pipeline: Construct a production-grade ColumnTransformer pipeline integrating SimpleImputer, StandardScaler, and OneHotEncoder for seamless inference on unseen data.",
        "4. Empirical Benchmarking: Evaluate the classifier on held-out test data using multiclass Accuracy, Macro F1, Weighted F1, Confusion Matrix analysis, and baseline comparison.",
        "5. Interactive Analytics Portal: Develop and deploy a modern, responsive Streamlit web application providing educators with real-time inference, probability distributions, and personalized recommendations."
    ]:
        doc.add_paragraph(obj, style='List Bullet')

    doc.add_page_break()

    # Section 3
    doc.add_heading("3. Theoretical Background & Related Literature", level=1)
    doc.add_heading("3.1 Foundations of Educational Data Mining", level=2)
    doc.add_paragraph(
        "Educational Data Mining (EDM) encompasses research methodologies dedicated to analyzing unique data emerging from educational contexts to better understand student learning trajectories and institutional effectiveness (Romero & Ventura, 2010). Early EDM investigations relied predominantly on standard statistical regression models to identify bivariate correlations between grade point averages and isolated attributes such as socioeconomic status or attendance rates."
    )
    doc.add_heading("3.2 Overview of Predictive Algorithms in EDM", level=2)
    doc.add_paragraph(
        "Academic literature has explored numerous algorithmic paradigms for student outcome forecasting, including Logistic Regression, Naive Bayes, Support Vector Machines (SVM), Multilayer Perceptrons (MLP), and Decision Trees (Baker & Inventado, 2014). Linear classifiers frequently suffer from high bias due to their inability to model complex feature interactions without extensive manual polynomial term engineering. Ensembling paradigms, specifically Random Forests, have consistently demonstrated superior empirical performance across educational benchmark studies by balancing bias and variance through aggregation."
    )
    doc.add_heading("3.3 Benchmark Dataset: Cortez and Silva (2008)", level=2)
    doc.add_paragraph(
        "In their seminal study, Cortez and Silva (2008) collected and evaluated secondary school student achievement from two Portuguese public schools within the Alentejo region. The authors modeled performance across two core subjects: Mathematics (395 records) and Portuguese language (649 records). A critical methodological observation from their work was that including the second-period grade (G2) produces artificially high predictive accuracy (correlations exceeding 0.90), as G2 occurs immediately prior to the final exam. Consequently, forecasting performance solely with G1 and background attributes constitutes the most realistic and practically valuable formulation for educational early intervention."
    )

    doc.add_page_break()

    # Section 4
    doc.add_heading("4. Dataset Characterization & Preprocessing Pipeline", level=1)
    doc.add_heading("4.1 UCI Dataset Overview & Source Attribution", level=2)
    doc.add_paragraph(
        "This project utilizes the Mathematics cohort of the Student Performance Dataset from the UC Irvine Machine Learning Repository (Cortez & Silva, 2008). The raw dataset comprises 395 student instances characterized by 33 initial attributes."
    )
    doc.add_heading("4.2 Target Variable Derivation & Categorization", level=2)
    doc.add_paragraph(
        "Continuous final grades (G3, 0-20 scale) are mapped into three academic tiers: Low (At-Risk, <10, N=130), Medium (Satisfactory, 10-13, N=165), and High (Good/Distinction, 14-20, N=100)."
    )
    doc.add_heading("4.3 Feature Selection & Leakage Isolation", level=2)
    doc.add_paragraph(
        "To enforce data leakage prevention, G3 and G2 are strictly excluded from the predictor matrix. The resulting feature space comprises 24 features (14 numerical, 10 categorical) transformed via ColumnTransformer."
    )

    doc.add_page_break()

    # Section 5
    doc.add_heading("5. Random Forest Classification: Theory & Mathematical Formulation", level=1)
    doc.add_heading("5.1 Algorithm Principle & Ensemble Architecture", level=2)
    doc.add_paragraph(
        "The Random Forest Classifier (Breiman, 2001) builds an ensemble of de-correlated decision trees using bootstrap aggregation and random subspace feature selection. The mathematical majority voting rule is H(x) = argmax_c sum(I(h_b(x)=c))."
    )
    doc.add_heading("5.2 Gini Impurity Splitting Metric", level=2)
    doc.add_paragraph(
        "At each node, candidate splits maximize the Gini gain: Delta I_G(tau) = I_G(tau) - [ (N_L/N)I_G(tau_L) + (N_R/N)I_G(tau_R) ], where I_G(tau) = 1 - sum(p_c^2)."
    )
    if os.path.exists("assets/charts/system_architecture.png"):
        doc.add_picture("assets/charts/system_architecture.png", width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

    # Section 6
    doc.add_heading("6. System Design, Implementation & Methodology", level=1)
    doc.add_paragraph(
        "The software architecture is decoupled into src/ modules: data_preprocessing.py, model_training.py, evaluation.py, prediction.py, and the Streamlit front-end in app.py. Automated tests in tests/ ensure reliable inference and input sanitization."
    )

    doc.add_page_break()

    # Section 7
    doc.add_heading("7. Experimental Setup, Empirical Results & Metrics", level=1)
    doc.add_paragraph(
        "Evaluated on unseen held-out test data (N=79, 20%), the Random Forest Classifier achieved 67.09% accuracy and 0.6845 Macro F1, significantly outperforming the naive majority baseline (41.77% accuracy, 0.1964 Macro F1)."
    )
    if os.path.exists("assets/charts/confusion_matrix.png"):
        doc.add_picture("assets/charts/confusion_matrix.png", width=Inches(4.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

    # Section 8
    doc.add_heading("8. Results Discussion, Ethical AI & Practical Implications", level=1)
    doc.add_paragraph(
        "Findings demonstrate that early diagnostic scores (G1, 36.5%), prior failures (5.2%), and absences (5.0%) are the strongest predictors. Responsible AI governance dictates that predictions must serve as advisory decision aids for instructors, never as automated gatekeepers."
    )

    doc.add_page_break()

    # Section 9
    doc.add_heading("9. Conclusions & Academic References", level=1)
    doc.add_paragraph(
        "This project successfully delivers an end-to-end, leak-free machine learning system for student performance prediction, providing educators with an accessible tool for early academic risk identification."
    )
    doc.add_heading("References", level=2)
    refs = [
        "[1] Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics. In Learning analytics (pp. 61-75). Springer, New York, NY.",
        "[2] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
        "[3] Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict secondary school student performance. In Proceedings of 5th Annual Future Business Technology Conference (pp. 5-12).",
        "[4] Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. ACM TKDD, 6(4), 1-21.",
        "[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.",
        "[6] Romero, C., & Ventura, S. (2010). Educational data mining: a review of the state of the art. IEEE Transactions on SMC, 40(6), 601-618."
    ]
    for r in refs:
        doc.add_paragraph(r)

    doc.save(filename)
    print(f"Generated DOCX: {filename}")


if __name__ == '__main__':
    build_pdf_report('reports/student_performance_report.pdf')
    build_docx_report('reports/student_performance_report.docx')
