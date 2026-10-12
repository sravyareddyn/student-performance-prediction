# Dataset Documentation: Student Performance Data Set

**Project:** Student Performance Prediction Using Machine Learning  
**Subject:** Fundamentals of Artificial Intelligence  

---

## 1. Official Dataset Attribution & Provenance

The primary dataset utilized in this project is the **Student Performance Data Set**, collected by **Paulo Cortez and Alice Silva** at the University of Minho, Portugal, and hosted on the **UC Irvine Machine Learning Repository**.

- **Official Repository URL:** [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Direct Archive Link:** `https://archive.ics.uci.edu/static/public/320/student+performance.zip`
- **Citation:**
  > Cortez, P., & Silva, A. M. G. (2008). *Using data mining to predict secondary school student performance.* In Proceedings of the 5th Annual Future Business Technology Conference (FUBUTEC 2008), Porto, Portugal, pp. 5–12. EUROSIS-ETI. ISBN 978-9077381-39-7.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 2. Dataset Files in this Directory

| File Name | Instances | Attributes | Description |
| :--- | :--- | :--- | :--- |
| `student-mat.csv` | 395 | 33 | Mathematics subject cohort (**Primary modeling dataset**) |
| `student-por.csv` | 649 | 33 | Portuguese language cohort |
| `student.txt` | — | — | Original author codebook and attribute descriptions |
| `student-merge.R` | — | — | Author-provided R script for identifying overlapping student records |

---

## 3. Data Acquisition Instructions

### Option A: Automatic Ingestion (Included in Pipeline)
The dataset files are pre-downloaded and stored directly within this `data/` folder for immediate out-of-the-box reproducibility.

### Option B: Fresh Download from Source (CLI)
To re-download and extract the official archive directly from the UCI repository:
```bash
python3 -c "
import urllib.request, zipfile, io, os
os.makedirs('data', exist_ok=True)
url = 'https://archive.ics.uci.edu/static/public/320/student+performance.zip'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    z.extractall('data')
with zipfile.ZipFile('data/student.zip') as z2:
    z2.extractall('data')
os.remove('data/student.zip')
print('UCI Dataset successfully extracted into data/')
"
```

### Option C: Synthetic Demonstration Fallback
If operating in an air-gapped, offline environment where network download is restricted and files are absent, execute the synthetic generation script (creates realistic synthetic educational records with realistic statistical distributions):
```bash
python3 -c "
import pandas as pd, numpy as np, os
np.random.seed(42)
n = 395
g1 = np.clip(np.random.normal(10.9, 3.3, n), 3, 20).round(1)
failures = np.random.choice([0, 1, 2, 3], size=n, p=[0.79, 0.13, 0.05, 0.03])
absences = np.clip(np.random.negative_binomial(2, 0.3, n), 0, 75)
g3 = np.clip(g1 + np.random.normal(-0.5, 2.0, n) - failures*1.5, 0, 20).round()
df = pd.DataFrame({
    'school': np.random.choice(['GP', 'MS'], n, p=[0.88, 0.12]),
    'sex': np.random.choice(['F', 'M'], n),
    'age': np.random.choice(range(15, 23), n, p=[0.21, 0.26, 0.25, 0.21, 0.05, 0.01, 0.005, 0.005]),
    'address': np.random.choice(['U', 'R'], n, p=[0.78, 0.22]),
    'famsize': np.random.choice(['GT3', 'LE3'], n, p=[0.71, 0.29]),
    'Pstatus': np.random.choice(['T', 'A'], n, p=[0.89, 0.11]),
    'Medu': np.random.choice([0, 1, 2, 3, 4], n, p=[0.01, 0.15, 0.26, 0.25, 0.33]),
    'Fedu': np.random.choice([0, 1, 2, 3, 4], n, p=[0.01, 0.21, 0.29, 0.25, 0.24]),
    'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n),
    'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n),
    'reason': np.random.choice(['home', 'reputation', 'course', 'other'], n),
    'guardian': np.random.choice(['mother', 'father', 'other'], n, p=[0.69, 0.23, 0.08]),
    'traveltime': np.random.choice([1, 2, 3, 4], n, p=[0.65, 0.27, 0.06, 0.02]),
    'studytime': np.random.choice([1, 2, 3, 4], n, p=[0.27, 0.50, 0.17, 0.06]),
    'failures': failures,
    'schoolsup': np.random.choice(['yes', 'no'], n, p=[0.13, 0.87]),
    'famsup': np.random.choice(['yes', 'no'], n, p=[0.61, 0.39]),
    'paid': np.random.choice(['yes', 'no'], n, p=[0.46, 0.54]),
    'activities': np.random.choice(['yes', 'no'], n, p=[0.51, 0.49]),
    'nursery': np.random.choice(['yes', 'no'], n, p=[0.79, 0.21]),
    'higher': np.random.choice(['yes', 'no'], n, p=[0.95, 0.05]),
    'internet': np.random.choice(['yes', 'no'], n, p=[0.83, 0.17]),
    'romantic': np.random.choice(['yes', 'no'], n, p=[0.33, 0.67]),
    'famrel': np.random.choice([1, 2, 3, 4, 5], n, p=[0.02, 0.05, 0.17, 0.50, 0.26]),
    'freetime': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.16, 0.40, 0.29, 0.10]),
    'goout': np.random.choice([1, 2, 3, 4, 5], n, p=[0.06, 0.26, 0.33, 0.22, 0.13]),
    'Dalc': np.random.choice([1, 2, 3, 4, 5], n, p=[0.70, 0.19, 0.07, 0.02, 0.02]),
    'Walc': np.random.choice([1, 2, 3, 4, 5], n, p=[0.38, 0.22, 0.20, 0.13, 0.07]),
    'health': np.random.choice([1, 2, 3, 4, 5], n, p=[0.12, 0.11, 0.23, 0.17, 0.37]),
    'absences': absences,
    'G1': g1,
    'G2': np.clip(g1 + np.random.normal(0, 1.5, n), 0, 20).round(),
    'G3': g3
})
os.makedirs('data', exist_ok=True)
df.to_csv('data/student-synthetic-demo.csv', sep=';', index=False)
print('Synthetic demonstration dataset generated at data/student-synthetic-demo.csv')
"
```
*(Note: As specified in project guidelines, synthetic data is strictly documented as demonstration-only and is never represented as empirical real-world collected data).*

---

## 4. Feature Taxonomy & Attribute Codebook

| Attribute | Data Type | Permissible Domain / Values | Domain Meaning | Included in Model? |
| :--- | :--- | :--- | :--- | :--- |
| `school` | Binary | `'GP'`, `'MS'` | Student's school ('GP': Gabriel Pereira, 'MS': Mousinho da Silveira) | **Excluded** (Institutional generalizability) |
| `sex` | Binary | `'F'`, `'M'` | Student's sex | **Included** |
| `age` | Integer | 15 to 22 | Student age in years | **Included** |
| `address` | Binary | `'U'`, `'R'` | Urban or Rural home address | **Included** |
| `famsize` | Binary | `'LE3'`, `'GT3'` | Family size (&le;3 or &gt;3) | **Included** |
| `Pstatus` | Binary | `'T'`, `'A'` | Parent cohabitation status ('T': together, 'A': apart) | **Included** |
| `Medu` | Ordinal | 0 to 4 | Mother education: 0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher | **Included** |
| `Fedu` | Ordinal | 0 to 4 | Father education: 0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher | **Included** |
| `Mjob` | Categorical | `'teacher'`, `'health'`, `'services'`, `'at_home'`, `'other'` | Mother's profession | **Included** |
| `Fjob` | Categorical | `'teacher'`, `'health'`, `'services'`, `'at_home'`, `'other'` | Father's profession | **Included** |
| `reason` | Categorical | `'home'`, `'reputation'`, `'course'`, `'other'` | Reason for attending current school | **Included** |
| `guardian` | Categorical | `'mother'`, `'father'`, `'other'` | Legal guardian | **Included** |
| `traveltime` | Ordinal | 1 to 4 | Transit time (1: &lt;15m, 2: 15-30m, 3: 30m-1h, 4: &gt;1h) | **Included** |
| `studytime` | Ordinal | 1 to 4 | Weekly study hours (1: &lt;2h, 2: 2-5h, 3: 5-10h, 4: &gt;10h) | **Included** |
| `failures` | Integer | 0 to 4 | Number of past course failures | **Included** |
| `schoolsup` | Binary | `'yes'`, `'no'` | Extra institutional educational support | **Included** |
| `famsup` | Binary | `'yes'`, `'no'` | Family educational support | **Included** |
| `paid` | Binary | `'yes'`, `'no'` | Extra paid classes within subject | **Included** |
| `activities` | Binary | `'yes'`, `'no'` | Extra-curricular activities participation | **Included** |
| `nursery` | Binary | `'yes'`, `'no'` | Attended preschool / nursery school | **Included** |
| `higher` | Binary | `'yes'`, `'no'` | Aspires to pursue higher university education | **Included** |
| `internet` | Binary | `'yes'`, `'no'` | Internet access at home | **Included** |
| `romantic` | Binary | `'yes'`, `'no'` | In a romantic relationship | **Included** |
| `famrel` | Ordinal | 1 to 5 | Quality of family relationship (1: very bad to 5: excellent) | **Included** |
| `freetime` | Ordinal | 1 to 5 | Free time after school (1: very low to 5: very high) | **Included** |
| `goout` | Ordinal | 1 to 5 | Social outings with friends (1: very low to 5: very high) | **Included** |
| `Dalc` | Ordinal | 1 to 5 | Workday alcohol consumption (1: very low to 5: very high) | **Included** |
| `Walc` | Ordinal | 1 to 5 | Weekend alcohol consumption (1: very low to 5: very high) | **Included** |
| `health` | Ordinal | 1 to 5 | Current health status (1: very bad to 5: very good) | **Included** |
| `absences` | Integer | 0 to 93 | Number of recorded school absences | **Included** |
| `G1` | Continuous | 0 to 20 | First period evaluation score (**Early diagnostic baseline**) | **Included** |
| `G2` | Continuous | 0 to 20 | Second period evaluation score | **Excluded (Data Leakage Protection)** |
| `G3` | Continuous | 0 to 20 | Final examination grade | **Excluded from X; used solely to derive target Y** |

---

## 5. Target Variable Derivation & Data Leakage Prevention

- **Target Construction:**
  Continuous grade $G3$ is categorized into 3 academic tiers:
  - `Low (At-Risk)`: $G3 < 10$ (130 students, 32.9%)
  - `Medium (Satisfactory)`: $10 \le G3 < 14$ (165 students, 41.8%)
  - `High (Good)`: $14 \le G3 \le 20$ (100 students, 25.3%)
- **Data Leakage Boundaries:**
  - $G3$ is strictly excluded from feature matrix $X$ because it is the ground-truth target source.
  - $G2$ (correlation 0.90 with $G3$) is strictly excluded because it represents late-stage information unavailable during early-warning intervention windows.
  - Only $G1$ (early formative checkpoint) and historical/environmental indicators are retained.
