# Video Demonstration Script & Recording Guide: Student Performance Prediction System
**Subject:** Fundamentals of Artificial Intelligence  
**Project:** Student Performance Prediction Using Machine Learning (Random Forest)  
**Target Duration:** 2 minutes 30 seconds to 3 minutes  
**Format:** Screen Recording with Voiceover (MP4, 1080p, 16:9)  

---

## 📋 Pre-Recording Setup & Checklist

Before recording your screen, complete these quick preparation steps:

1. **Activate Virtual Environment & Launch Dashboard:**
   ```bash
   cd /Users/sravya/Desktop/SEM7/AI/project
   ./venv/bin/streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.
2. **Browser Layout:**
   - Set browser zoom to 100% or 110% for crisp readability.
   - Collapse bookmarks toolbar and browser tabs.
3. **Screen Recording Tool (macOS):**
   - Press **`Cmd + Shift + 5`** to open the native macOS Screen Recording toolbar.
   - Select **Record Entire Screen** or **Record Selected Portion** (drag over your browser window).
   - In **Options**, select your microphone (e.g., Built-in Microphone).
   - Click **Record**.
4. **Audio Check:**
   - Speak clearly at a moderate, confident pace.

---

## 🎬 Video Demonstration Script (Timestamp Breakdown)

### 0:00 – 0:30 | Scene 1: Introduction & Executive Problem Context
**Screen:** Show the **Overview** page (`http://localhost:8501`), displaying the top metric cards and executive summary.  
**Action:** Move cursor smoothly over the header banner and metric cards (67.1% Test Accuracy, 0.685 Macro F1, +25.3% over baseline).  
**Spoken Narration:**  
> *"Hello everyone! In this video, I am demonstrating our project for **Fundamentals of Artificial Intelligence**: **Student Performance Prediction Using Machine Learning**.*  
> *In educational institutions, identifying students who need academic support typically happens too late—after final examinations have concluded.*  
> *Our project builds a proactive machine learning decision-support framework powered by an optimized **Random Forest Classifier**. By evaluating early-term diagnostic scores alongside attendance and socio-demographic indicators, our system predicts student performance across three tiers: Low (At-Risk), Medium (Satisfactory), and High Performance—enforcing strict data leakage prevention to make early intervention possible."*

---

### 0:30 – 1:05 | Scene 2: Dataset Explorer & Distribution Insights
**Screen:** Click on **"📊 Dataset Explorer"** in the sidebar.  
**Action:** Scroll down through the sample dataframe (395 records, 33 original attributes), the descriptive statistics table, and pause briefly at the two charts: Final Grade distribution and Study Time boxplot.  
**Spoken Narration:**  
> *"Here in the **Dataset Explorer**, we can inspect our data, sourced from the verified UCI Machine Learning Repository.*  
> *The dataset tracks 395 secondary students. As we see in the summary statistics, there are zero missing values.*  
> *The grade distribution shows a mean of 10.4 out of 20, with 10 serving as the official threshold for course passing.*  
> *Our pipeline automatically derives the 3 target categories while strictly excluding final and intermediate exam grades from the feature matrix to eliminate target leakage."*

---

### 1:05 – 1:55 | Scene 3: Live Interactive Performance Prediction
**Screen:** Click on **"🎯 Predict Performance"** in the sidebar.  
**Action:**  
1. Click **"Profile A: High Achiever"** preset to show instant auto-filling.  
2. Then click **"Profile C: At-Risk Student"** preset.  
3. Show the form parameters: First period score ($G1 = 6.0$), Weekly Study Time ($<2$ hours), Past failures ($2$), Absences ($16$).  
4. Click the purple button: **"🔮 Predict Performance Tier"**.  
5. Show the resulting output card: Red badge `Low (At-Risk)`, confidence score, class probability bars, and pedagogical recommendations.  
**Spoken Narration:**  
> *"Now let's test the interactive prediction interface.*  
> *The system provides convenient presets or full manual adjustment of 24 input features.*  
> *Let's test an at-risk profile: a student with an early diagnostic score of 6 out of 20, less than 2 hours of weekly study, 2 prior failures, and 16 absences.*  
> *When we click 'Predict Performance Tier', our cached Random Forest model evaluates the student in milliseconds.*  
> *The model classifies the student as **Low (At-Risk)** with high confidence, accompanied by calibrated class probabilities across all three categories.*  
> *Importantly, the system automatically generates targeted pedagogical interventions—such as enrolling in peer tutoring, scheduling weekly check-ins, and monitoring attendance.*  
> *Notice our responsible AI disclaimer: predictions are probabilistic guides for teachers, not automated punitive decisions."*

---

### 1:55 – 2:35 | Scene 4: Empirical Model Evaluation & Findings
**Screen:** Click on **"📈 Model Evaluation"** in the sidebar.  
**Action:** Scroll through the metric cards, zoom in on the **Confusion Matrix Heatmap**, and showcase the **Top Informative Features** bar chart.  
**Spoken Narration:**  
> *"Next, let's examine our empirical evaluation conducted on an unseen 20% test partition of 79 students.*  
> *Our Random Forest model achieved a **test accuracy of 67.1%** and a **Macro F1-score of 0.685**, outperforming the baseline majority classifier by over 25 percentage points.*  
> *The confusion matrix demonstrates strong precision—over 82%—when identifying high-performing students, while capturing over 57% of at-risk students before final exams.*  
> *Looking at feature importances derived from Gini impurity, early diagnostic scores ($G1$), past course failures, and absenteeism are the top three drivers of student trajectory, aligning closely with educational research."*

---

### 2:35 – 2:55 | Scene 5: Conclusion & Wrap-Up
**Screen:** Click on **"ℹ️ About Project"** or return to **"🏠 Overview"**.  
**Action:** Conclude calmly and display the project summary.  
**Spoken Narration:**  
> *"In summary, this project delivers a functional, leak-free, empirically verified machine learning solution that bridges algorithmic theory with real-world educational intervention.*  
> *All source code, automated test suites, documentation, and the full 10-page report are packaged in our GitHub repository.*  
> *Thank you very much for watching!"*

---

## 💾 Post-Recording Steps

1. Stop recording (press **`Cmd + Ctrl + Esc`** or click the stop icon in the menu bar).
2. Save the recording inside `demo/` as:
   ```
   demo/student_performance_demo.mp4
   ```
3. To compress or convert the video for upload submission:
   ```bash
   ffmpeg -i demo/student_performance_demo.mov -vcodec h264 -acodec aac demo/student_performance_demo.mp4
   ```
4. Verify playback in QuickTime Player or VLC.
