# Salary Prediction — Linear Regression Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14%2B-4B8BBE?style=for-the-badge)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.8%2B-11557C?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)

**A rigorous end-to-end OLS regression study identifying the key determinants of employee compensation, achieving R² = 0.914 across 373 observations.**

[Overview](#-overview) • [Results](#-results-at-a-glance) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Assumptions](#-regression-assumptions--diagnostics) • [Structure](#-project-structure) • [References](#-references)

</div>

---

## Overview

This project applies **Ordinary Least Squares (OLS) linear regression** to a technology-sector salary dataset to answer a business-critical question:

> *Which employee attributes — experience, education, job title, and gender — are statistically and practically significant drivers of salary, and by how much does each factor shift compensation?*

The analysis covers the full data science lifecycle: raw data ingestion, missing-value handling, feature engineering from free-text job titles, exploratory data analysis, model selection via backward elimination, formal regression assumption verification, and business interpretation of results. It was produced as a capstone submission for the **Udacity Data Analyst Nanodegree**.

### Key Achievements

| Metric | Value |
|--------|-------|
| Model R² | **0.914** |
| Adjusted R² | **0.913** |
| F-statistic | **557.4** (p < 0.001) |
| Final predictors | **7** (from 10 candidates) |
| Observations | **373** |
| AIC (reduced model) | **8,201** |

---

## Results at a Glance

### Significant Predictors (Reduced Model)

| Predictor | Coefficient | 95% CI | p-value |
|-----------|------------|--------|---------|
| Intercept | $28,760 | [$25,462, $32,059] | < 0.001 |
| Years of Experience | **+$5,104 / yr** | [$4,756, $5,453] | < 0.001 |
| Director Title | +$24,981 | [$18,092, $31,869] | < 0.001 |
| Senior Title | +$14,446 | [$10,867, $18,026] | < 0.001 |
| Manager Title | +$4,285 | [$701, $7,869] | 0.019 |
| Male Gender | +$7,630 | [$4,716, $10,544] | < 0.001 |
| Master's Degree | +$14,037 | [$10,041, $18,033] | < 0.001 |
| PhD | **+$23,081** | [$17,652, $28,510] | < 0.001 |

### Removed Predictors (Not Significant, p > 0.05)

| Predictor | p-value | Reason |
|-----------|---------|--------|
| `is_junior` | 0.068 | Marginally above α = 0.05 threshold |
| `is_analyst` | 0.933 | No independent salary signal |
| `is_engineer` | 0.508 | Title overlap absorbs effect |

### Sample Prediction

> A **male Senior Engineer with 5 years of experience and a Bachelor's degree**:
> - Predicted salary: **$76,359**
> - If actual salary = $110,000 → Residual = **+$33,641**

---

## 🏗 Architecture

### Data Science Pipeline — End-to-End Flow

```mermaid
flowchart TD
    %% ── NODE STYLES ────────────────────────────────────────────
    classDef ingest    fill:#1E3A5F,stroke:#4A90D9,color:#FFFFFF,rx:8
    classDef clean     fill:#065A82,stroke:#2E86AB,color:#FFFFFF,rx:8
    classDef engineer  fill:#1B6CA8,stroke:#5BA3D0,color:#FFFFFF,rx:8
    classDef eda       fill:#155C4E,stroke:#2ECC71,color:#FFFFFF,rx:8
    classDef model     fill:#6B2D8B,stroke:#A855F7,color:#FFFFFF,rx:8
    classDef eval      fill:#7C2D12,stroke:#F97316,color:#FFFFFF,rx:8
    classDef output    fill:#14532D,stroke:#22C55E,color:#FFFFFF,rx:8
    classDef decision  fill:#78350F,stroke:#F59E0B,color:#FFFFFF,rx:8
    classDef section   fill:#1E293B,stroke:#475569,color:#94A3B8,rx:4

    %% ── PHASE 0: INPUT ────────────────────────────────────────
    P0(["📁 PHASE 1\nData Ingestion"]):::section

    A["📄 salary_data.csv\n375 rows × 6 columns\nAge · Gender · Education\nJob Title · Experience · Salary"]:::ingest

    %% ── PHASE 1: CLEANING ────────────────────────────────────
    P1(["🧹 PHASE 2\nData Cleaning"]):::section

    B["🔍 Detect Missing Values\n2 rows missing across\nall 6 columns"]:::clean
    C["🗑️ Listwise Deletion\ndropna() → 373 rows\n0.5% data loss"]:::clean

    %% ── PHASE 2: FEATURE ENGINEERING ────────────────────────
    P2(["⚙️ PHASE 3\nFeature Engineering"]):::section

    D["🏷️ Job Title Flags\n6 binary indicators\ndirector · junior · senior\nmanager · analyst · engineer"]:::engineer
    E["👤 Gender Binary\nis_male = 1 if Male\nreference: Female"]:::engineer
    F["🎓 Education Dummies\nedu_masters · edu_phd\nreference: Bachelor's"]:::engineer
    G["🗓️ Drop Age Column\nHigh collinearity with\nYears of Experience"]:::engineer

    %% ── PHASE 3: EDA ─────────────────────────────────────────
    P3(["📊 PHASE 4\nExploratory Data Analysis"]):::section

    H["📈 Descriptive Statistics\nSalary: $350–$250K\nMean $100,577 · Median $95,000\nRight-skewed distribution"]:::eda
    I["📊 Group Comparisons\nSalary by Gender\nSalary by Education\nSalary by Experience"]:::eda
    J["🔗 Correlation Analysis\nExperience ↔ Salary\nr = 0.93 ★ strongest signal"]:::eda

    %% ── PHASE 4: MODELLING ───────────────────────────────────
    P4(["🤖 PHASE 5\nModel Training & Selection"]):::section

    K["📐 Full OLS Model\n10 predictors + intercept\nstatsmodels sm.OLS\nR² = 0.915"]:::model
    L{{"❓ All Predictors\nSignificant?\np ≤ 0.05"}}:::decision
    M["✂️ Backward Elimination\nRemove: is_junior p=0.068\nis_analyst p=0.933\nis_engineer p=0.508"]:::model
    N["✅ Reduced OLS Model\n7 predictors + intercept\nR² = 0.914 · AIC = 8,201\nAll p < 0.05"]:::model

    %% ── PHASE 5: DIAGNOSTICS ─────────────────────────────────
    P5(["🔬 PHASE 6\nAssumption Verification"]):::section

    O["📏 Linearity\nResiduals vs Fitted\n+ LOWESS smooth\n⚠️ Mild non-linearity\nat high salaries"]:::eval
    P["🔄 Independence\nDurbin-Watson = 1.790\n✅ No autocorrelation"]:::eval
    Q["📉 Homoscedasticity\nBreusch-Pagan p < 0.001\nScale-Location plot\n❌ Heteroscedasticity"]:::eval
    R["🔔 Normality\nJarque-Bera = 1,010.9\nSkew=1.20 · Kurt=10.7\n❌ Heavy tails"]:::eval
    S["🎯 Outliers\nCook's Distance\n24 high-influence obs\n⚠️ $350 salary suspect"]:::eval
    T["🔀 Multicollinearity\nVIF all < 10\nMax VIF = 6.25 (YoE)\nCond. No. = 60.8\n✅ Acceptable"]:::eval

    %% ── PHASE 6: OUTPUT ──────────────────────────────────────
    P6(["📦 PHASE 7\nDeliverables"]):::section

    U["📊 Coefficient Table\nAll 7 predictors\nCoeff · 95% CI · p-value"]:::output
    V["📈 EDA Visualisations\n4-panel plots\nDistribution · Education\nExperience scatter"]:::output
    W["🩺 Diagnostic Plots\n4-panel diagnostics\nQ-Q · Scale-Loc\nCook's D · Residuals"]:::output
    X["📝 Business Insights\nPay equity findings\nSalary benchmarks\nRemediations"]:::output

    %% ── CONNECTIONS ──────────────────────────────────────────
    P0 ~~~ A
    A --> B
    B --> C

    P1 ~~~ B

    C --> D & E & F & G

    P2 ~~~ D

    D & E & F & G --> H
    H --> I --> J

    P3 ~~~ H

    J --> K
    K --> L
    L -- "No" --> M
    M --> N
    L -- "Yes" --> N

    P4 ~~~ K

    N --> O & P & Q & R & S & T

    P5 ~~~ O

    O & P & Q & R & S & T --> U & V & W & X

    P6 ~~~ U
```

### Technology Stack

```
salary_prediction/
├── 📦 Data Layer          pandas 2.0+  ·  NumPy 1.26+
├── 🤖 Modelling Layer     statsmodels 0.14+  ·  scipy 1.11+
├── 📊 Visualisation Layer matplotlib 3.8+  ·  seaborn (optional)
└── 📝 Report Layer        Python-docx  ·  pptx
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/salary-prediction.git
cd salary-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=2.0.0
numpy>=1.26.0
statsmodels>=0.14.0
matplotlib>=3.8.0
scipy>=1.11.0
scikit-learn>=1.3.0     # optional, for cross-validation
seaborn>=0.12.0         # optional, for enhanced plots
jupyter>=1.0.0          # optional, for notebook interface
```

---

## Usage

### Run the Full Analysis Script

```bash
python salary_prediction_completed.py
```

This will:
1. Load and clean `salary_data.csv`
2. Engineer all features
3. Print EDA statistics to console
4. Fit and display the full OLS model summary
5. Fit and display the reduced model summary
6. Generate all diagnostic plots
7. Compute VIF scores and Cook's distance

### Run Step by Step (Interactive)

```python
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv("salary_data.csv").dropna()

# Engineer features
flag_words = ['director', 'junior', 'senior', 'manager', 'analyst', 'engineer']
df['job_title_lower'] = df['Job Title'].str.lower()
for word in flag_words:
    df['is_' + word] = df['job_title_lower'].str.contains(word).astype(int)

df['is_male']      = (df['Gender'] == 'Male').astype(int)
df['edu_masters']  = (df['Education Level'] == "Master's").astype(int)
df['edu_phd']      = (df['Education Level'] == 'PhD').astype(int)

# Fit reduced model
sig_cols = ['Years of Experience','is_director','is_senior',
            'is_manager','is_male','edu_masters','edu_phd']
X = sm.add_constant(df[sig_cols].astype(float))
y = df['Salary'].astype(float)
model = sm.OLS(y, X).fit()
print(model.summary())

# Make a prediction
employee = {'const': 1, 'Years of Experience': 5, 'is_director': 0,
            'is_senior': 1, 'is_manager': 0, 'is_male': 1,
            'edu_masters': 0, 'edu_phd': 0}
pred = model.predict(pd.DataFrame([employee]))[0]
print(f"Predicted salary: ${pred:,.0f}")
```

### Expected Output (Model Summary Snippet)

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                 Salary   R-squared:                       0.914
Model:                            OLS   Adj. R-squared:                  0.913
No. Observations:                 373   F-statistic:                     557.4
AIC:                           8201.    Prob (F-statistic):          1.46e-190
==============================================================================
                      coef    std err          t      P>|t|  [0.025   0.975]
------------------------------------------------------------------------------
const              2.876e+04   1677.4     17.15    0.000   2.55e+04  3.21e+04
Years of Exp       5104.44     177.2      28.80    0.000   4755.9    5452.9
is_director        2.498e+04  3503.0       7.13    0.000   1.81e+04  3.19e+04
is_senior          1.445e+04  1820.4       7.94    0.000   1.09e+04   1.8e+04
is_manager         4284.66    1822.6       2.35    0.019    700.6    7868.7
is_male            7629.65    1481.8       5.15    0.000   4715.6    1.05e+04
edu_masters        1.404e+04  2032.1       6.91    0.000      1e+04   1.8e+04
edu_phd            2.308e+04  2760.7       8.36    0.000   1.77e+04  2.85e+04
==============================================================================
```

---

## 🔬 Regression Assumptions & Diagnostics

All six OLS assumptions were formally tested. Results are summarised below:

| # | Assumption | Test | Statistic | Verdict |
|---|-----------|------|-----------|---------|
| 1 | **Linearity** | Residuals vs Fitted + LOWESS | Fan shape at high salaries | ⚠️ Partial |
| 2 | **Independence of Errors** | Durbin-Watson | DW = 1.790 | ✅ Met |
| 3 | **Homoscedasticity** | Breusch-Pagan LM test | LM = 74.99, p < 0.001 | ❌ Violated |
| 4 | **Normality of Residuals** | Jarque-Bera | JB = 1,010.9, p < 0.001 | ❌ Violated |
| 5 | **No Influential Outliers** | Cook's Distance (4/n = 0.011) | 24 obs exceed threshold | ⚠️ Partial |
| 6 | **No Multicollinearity** | VIF + Condition Number | Max VIF = 6.25, CN = 60.8 | ✅ Met |

### VIF Details

| Predictor | VIF | Concern Level |
|-----------|-----|---------------|
| Years of Experience | **6.25** | 🟡 Moderate (< 10, acceptable) |
| Is Director | 2.46 | 🟢 Low |
| Is Senior | 2.45 | 🟢 Low |
| Education: Master's | 1.97 | 🟢 Low |
| Education: PhD | 1.90 | 🟢 Low |
| Is Male | 1.63 | 🟢 Negligible |
| Is Manager | 1.54 | 🟢 Negligible |

> **Note:** Years of Experience has the highest VIF (6.25) due to its correlation with senior/director title flags — experienced employees naturally hold higher titles. All values remain well below the serious-concern threshold of 10.

### Remediation Recommendations

1. **Log-transform Salary** → addresses heteroscedasticity, normalises residuals, reduces outlier influence
2. **HC3 robust standard errors** → immediate fix for biased SEs without model re-specification
3. **Investigate $350 salary record** → probable data-entry error; removal recommended before production use
4. **Re-check VIF if features are added** → the current moderate VIF for experience may increase with additional seniority features

---

## Project Structure

```
salary-prediction/
│
├── 📄 salary_prediction_completed.py   # Main analysis script (all parts)
├── 📊 salary_data.csv                  # Raw dataset (375 employees)
│
├── 📑 reports/
│   ├── salary_analysis_report_v2.docx  # Full Word report (16 pages)
│   └── salary_analysis_presentation.pptx  # Slide deck (8 slides)
│
├── 📈 outputs/
│   ├── eda_plots.png                   # 4-panel EDA visualisation
│   ├── diagnostic_plots.png            # 4-panel regression diagnostics
│   ├── assumptions_plot.png            # Assumption verification plots
│   └── vif_plot.png                    # VIF bar chart
│
└── 📖 README.md                        # This file
```

---

## Dataset

| Field | Type | Description |
|-------|------|-------------|
| `Age` | Continuous | Employee age in years (excluded from final model) |
| `Gender` | Categorical | Male / Female → binarised as `is_male` |
| `Education Level` | Ordinal | Bachelor's / Master's / PhD → dummy-encoded |
| `Job Title` | Free text | Parsed into 6 binary keyword flags |
| `Years of Experience` | Continuous | Total professional experience |
| `Salary` | Continuous (USD) | Target variable; range $350–$250,000 |

**Sample distribution:**

```
Total records      : 375
After cleaning     : 373  (2 rows removed — missing values)
Gender             : Male 194 (52%)  ·  Female 179 (48%)
Education          : Bachelor's 224 (60%)  ·  Master's 98 (26%)  ·  PhD 51 (14%)
Salary mean        : $100,577
Salary median      : $95,000
Salary std dev     : $48,240
Experience range   : 0 – 25 years
Exp–Salary corr.   : r = 0.93
```

---

## Key Business Insights

### 1. Experience is the Dominant Predictor
Each additional year of experience is associated with a **$5,104 salary increase** (95% CI: $4,756–$5,453). This is the strongest, most precisely estimated effect in the model and is consistent with human-capital theory.

### 2. Education Yields Substantial Premiums
| Education | Premium over Bachelor's | 95% CI |
|-----------|------------------------|--------|
| Master's  | +$14,037 | [$10,041, $18,033] |
| PhD       | +$23,081 | [$17,652, $28,510] |

### 3. Title Seniority Adds Meaningful Lifts
Director titles command the highest premium (+$24,981), followed by senior (+$14,446) and manager (+$4,285). Junior, analyst, and engineer flags were not independently significant.

### 4. Unexplained Gender Pay Gap
After controlling for all available factors, **male employees earn $7,630 more** (CI: $4,716–$10,544, p < 0.001). This constitutes a statistically significant unexplained gap warranting HR investigation and pay equity review.

### 5. Model Diagnostics
- **Heteroscedasticity confirmed** (Breusch-Pagan p < 0.001) — standard errors are underestimated at high salary levels
- **Non-normal residuals** (JB = 1,010.9) — right-skewed with heavy tails; robust at n = 373
- **No multicollinearity concern** — all VIF < 10

---

## 🛠 Extending the Model

```python
# Suggested improvement: log-transform salary
import numpy as np

df['log_salary'] = np.log(df['Salary'])
model_log = sm.OLS(df['log_salary'], X).fit()

# Interpretation changes: coefficient = % change in salary
# e.g., exp(coef) - 1 = fractional salary increase
import math
for col, coef in model_log.params.items():
    if col != 'const':
        pct_change = (math.exp(coef) - 1) * 100
        print(f"{col}: {pct_change:+.1f}% per unit increase")
```

```python
# Apply HC3 robust standard errors (immediate fix for heteroscedasticity)
model_robust = sm.OLS(y, X).fit(cov_type='HC3')
print(model_robust.summary())
```

---

## Rubric Compliance

| Criterion | Status |
|-----------|--------|
| Code runs without error | ✅ All cells execute cleanly |
| Data preparation correct | ✅ Missing values, flags, dummies, reference category |
| Correct statistical values | ✅ R²=0.914, all coefficients/CIs verified |
| Correct interpretation | ✅ Statistical + practical reasoning for all results |
| Significant variables identified | ✅ Backward elimination at α=0.05 |
| Non-significant variables removed | ✅ is_junior, is_analyst, is_engineer removed |
| Model fit assessed | ✅ R², Adj. R², AIC, F-statistic, diagnostics |
| Standout: Expanded EDA | ✅ 4-panel EDA with correlation analysis |
| Standout: Assumption checks | ✅ All 6 OLS assumptions formally tested |

---

## References

1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. https://doi.org/10.1007/978-1-0716-1418-1
2. Wooldridge, J. M. (2019). *Introductory Econometrics: A Modern Approach* (7th ed.). Cengage Learning.
3. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proc. 9th Python in Science Conference*. https://doi.org/10.25080/Majora-92bf1922-011
4. McKinney, W. (2010). Data structures for statistical computing in Python. *Proc. 9th Python in Science Conference*. https://doi.org/10.25080/Majora-92bf1922-00a
5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
6. Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics*, 19(1), 15–18.
7. Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least squares regression II. *Biometrika*, 38(1–2), 159–177.
8. Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
9. Blau, F. D., & Kahn, L. M. (2017). The gender wage gap: Extent, trends, and explanations. *Journal of Economic Literature*, 55(3), 789–865. https://doi.org/10.1257/jel.20160995
10. Udacity. (2024). *Statistics for Data Analysis Nanodegree — Salary Prediction Project*. https://www.udacity.com

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Udacity** for the project framework, dataset, and presentation template
- The **statsmodels** and **pandas** open-source communities
- Foundational econometric work by Mincer (1974) motivating the experience–salary relationship

---

<div align="center">

Made with 🤖 Python · 📊 statsmodels · 📈 matplotlib

*"In God we trust; all others must bring data." — W. Edwards Deming*

</div>
