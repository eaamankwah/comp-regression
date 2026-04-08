#!/usr/bin/env python
# coding: utf-8

# # Salary Prediction
#
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Descriptive Statistics](#descriptive)
# - [Part II - Regression](#regression)
# - [Part III - Interpret Results](#interpret)
#
#
# <a id='intro'></a>
# ### Introduction
#
# Linear Regression is very commonly performed by data analysts and data scientists. For this project,
# you will be working to understand the results of a Linear Regression model associated with salaries.
# Your goal is to work through this notebook to understand what variables are related to salary,
# and how exactly they are related.

# In[1]:

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

random.seed(0)
np.random.seed(0)


# <a id='descriptive'></a>
# #### Part I - Descriptive Statistics

# `1.a)` Read in the salary_data.csv data. **This question is completed for you**

# In[2]:

df = pd.read_csv('salary_data.csv')
df.head()


# `b)` Find the number of rows in the dataset.

# In[3]:

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
# Output: 375 rows, 6 columns


# `c)` Do any of the rows have missing values? If so, handle them.

# In[4]:

print("Missing values per column:")
print(df.isnull().sum())
# 2 missing values per column (same 2 rows across all columns)

# Drop rows with any missing values (listwise deletion)
df = df.dropna()
print(f"\nRows after dropping missing: {df.shape[0]}")  # 373


# In[5]:

# Confirm no missing values remain
print("Missing values after cleaning:")
print(df.isnull().sum())


# `d)` How many employees are in each Education Level? Build a bar chart.

# In[6]:

print("Employee count by Education Level:")
print(df['Education Level'].value_counts())


# In[7]:

# Bar chart of results
df['Education Level'].value_counts().plot(kind='bar')
plt.title('Number of Visits From Each Education Level')
plt.ylabel('Count of Visits')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# `e)` What are the possible values for Salary? What does the distribution look like?

# In[8]:

print("Salary statistics:")
print(df['Salary'].describe())
print(f"\nMin salary: ${df['Salary'].min():,.0f}")
print(f"Max salary: ${df['Salary'].max():,.0f}")
print(f"Mean salary: ${df['Salary'].mean():,.0f}")
print(f"Median salary: ${df['Salary'].median():,.0f}")


# In[9]:

plt.figure(figsize=(10, 5))
plt.hist(df['Salary'] / 1000, bins=30, color='steelblue', edgecolor='white')
plt.title('Distribution of Salary')
plt.xlabel('Salary ($000s)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
# Distribution is right-skewed; most salaries cluster below $100k but a long right tail extends to $250k


# <a id='regression'></a>
# #### Part II - Regression

# `1.a)` Consider average salary by gender. **This question is completed for you (corrected)**

# In[10]:

print("Average salary by gender:")
print(df.groupby('Gender')['Salary'].mean())
# Male: ~$103,868 | Female: ~$97,011
# Males earn on average ~$6,857 more; likely significant (confirmed in regression)


# `b)` Consider average salary by education level.

# In[11]:

print("Average salary by education level:")
edu_salary = df.groupby('Education Level')['Salary'].mean()
print(edu_salary)
# Bachelor's: $74,756 | Master's: $129,796 | PhD: $157,843
# Clear monotonic increase — strong evidence education is positively associated with salary


# `c)` Consider average salary by years of experience.

# In[12]:

corr = df['Years of Experience'].corr(df['Salary'])
print(f"Correlation between Years of Experience and Salary: {corr:.4f}")
# r = 0.9303 — very strong positive linear relationship

plt.figure(figsize=(8, 5))
plt.scatter(df['Years of Experience'], df['Salary'] / 1000, alpha=0.4, color='steelblue', s=20)
m, b = np.polyfit(df['Years of Experience'], df['Salary'] / 1000, 1)
x_line = np.linspace(df['Years of Experience'].min(), df['Years of Experience'].max(), 100)
plt.plot(x_line, m * x_line + b, 'r-', linewidth=2)
plt.title(f'Years of Experience vs Salary (r = {corr:.2f})')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($000s)')
plt.tight_layout()
plt.show()


# `d)` Create job title flag columns. **This question is completed for you**

# In[13]:

flag_words = ['director', 'junior', 'senior', 'manager', 'analyst', 'engineer']
df['job_title'] = df['Job Title'].str.lower()

for word in flag_words:
    df['is_' + word] = df['job_title'].str.contains(word)

df = df.drop('job_title', axis=1)
df = df.drop('Job Title', axis=1)


# `e)` Create a flag for gender where 1 = male, 0 = not male.

# In[14]:

df['is_male'] = (df['Gender'] == 'Male').astype(int)
df = df.drop('Gender', axis=1)
print("Gender flag created. Value counts:")
print(df['is_male'].value_counts())


# `f)` Use statsmodels to fit a linear model predicting salary.

# In[15]:

# Convert bool job title flags to int
for word in flag_words:
    df['is_' + word] = df['is_' + word].astype(int)

# Create education dummies (Bachelor's = reference category, omitted)
df['edu_masters'] = (df['Education Level'] == "Master's").astype(int)
df['edu_phd'] = (df['Education Level'] == "PhD").astype(int)
df = df.drop('Education Level', axis=1)
df = df.drop('Age', axis=1)  # Age highly collinear with experience

# Define feature matrix and target
feature_cols = ['Years of Experience', 'is_director', 'is_junior', 'is_senior',
                'is_manager', 'is_analyst', 'is_engineer', 'is_male',
                'edu_masters', 'edu_phd']
X = df[feature_cols].astype(float)
X = sm.add_constant(X)
y = df['Salary'].astype(float)

# Fit OLS model
model_full = sm.OLS(y, X).fit()
print(model_full.summary())


# <a id='interpretation'></a>
# ### Part III - Interpret Results

# `1.` Identify significant features; fit reduced model.

# In[16]:

# Variables with p > 0.05 in the full model:
# is_junior (p=0.068), is_analyst (p=0.933), is_engineer (p=0.508)
# These are not statistically significant at alpha=0.05 and are removed.

significant_cols = ['const', 'Years of Experience', 'is_director', 'is_senior',
                    'is_manager', 'is_male', 'edu_masters', 'edu_phd']

X_reduced = X[significant_cols]
model = sm.OLS(y, X_reduced).fit()
print(model.summary())
# Reduced model R² = 0.914, essentially identical to full model (0.915)
# AIC improves from 8203 to 8201 — confirms removing insignificant vars was correct


# `a)` With each additional year of experience, what is the expected change in salary?

# **Answer:**
# Coefficient = +$5,104 per year
# 95% CI: [$4,756, $5,453]
# For each additional year of experience, salary is expected to increase by $5,104,
# holding all other variables constant. We are 95% confident this increase is between
# $4,756 and $5,453 (p < 0.001).

# In[17]:

yoe_coef = model.params['Years of Experience']
yoe_ci = model.conf_int().loc['Years of Experience']
print(f"Years of Experience coefficient: ${yoe_coef:,.0f}")
print(f"95% CI: [${yoe_ci[0]:,.0f}, ${yoe_ci[1]:,.0f}]")


# `b)` Expected difference in salary: senior title vs no title indication?

# **Answer:**
# Coefficient for is_senior = +$14,450
# An employee with a senior title earns approximately $14,450 more than an otherwise
# identical employee with none of the flagged title indicators.

# In[18]:

senior_coef = model.params['is_senior']
senior_ci = model.conf_int().loc['is_senior']
print(f"Senior title premium: ${senior_coef:,.0f}")
print(f"95% CI: [${senior_ci[0]:,.0f}, ${senior_ci[1]:,.0f}]")


# `c)` Expected difference in salary: PhD vs Bachelor's? 95% CI?

# **Answer:**
# Coefficient for edu_phd = +$23,080
# 95% CI: [$17,652, $28,510]
# An employee with a PhD is expected to earn $23,080 more than an equivalent employee
# with a Bachelor's degree. We are 95% confident this difference is between $17,652 and
# $28,510 (p < 0.001).

# In[19]:

phd_coef = model.params['edu_phd']
phd_ci = model.conf_int().loc['edu_phd']
print(f"PhD salary premium over Bachelor's: ${phd_coef:,.0f}")
print(f"95% CI: [${phd_ci[0]:,.0f}, ${phd_ci[1]:,.0f}]")


# `d)` Predicted salary: male, 5 years experience, senior engineer, bachelor's degree.

# **Answer:** $76,359

# In[20]:

pred_input = pd.DataFrame({
    'const': [1],
    'Years of Experience': [5],
    'is_director': [0],
    'is_senior': [1],     # Senior Engineer → is_senior = 1
    'is_manager': [0],
    'is_male': [1],
    'edu_masters': [0],
    'edu_phd': [0]        # Bachelor's → edu_masters=0, edu_phd=0
})
predicted_salary = model.predict(pred_input)[0]
print(f"Predicted salary: ${predicted_salary:,.0f}")
# Note: is_engineer was dropped as not significant, so only is_senior applies here


# `e)` If actual salary = $110,000, what is the residual?

# **Answer:** +$33,641 (model under-predicts)

# In[21]:

actual_salary = 110000
residual = actual_salary - predicted_salary
print(f"Predicted: ${predicted_salary:,.0f}")
print(f"Actual:    ${actual_salary:,.0f}")
print(f"Residual:  ${residual:,.0f}")
# A positive residual means the model under-predicts this employee's salary.
# This could reflect unmeasured factors like performance, company size, or location.


# `f)` How well does the model fit? Metrics and diagnostic plots.

# In[22]:

print(f"R-squared:       {model.rsquared:.4f}")
print(f"Adj. R-squared:  {model.rsquared_adj:.4f}")
print(f"F-statistic:     {model.fvalue:.1f}  (p = {model.f_pvalue:.2e})")
print(f"AIC:             {model.aic:.1f}")
print(f"Observations:    {int(model.nobs)}")

# The model explains 91.4% of variance in salary — an excellent fit for cross-sectional
# salary data. All 7 predictors are statistically significant (p < 0.05).

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Regression Diagnostics', fontsize=14, fontweight='bold')

fitted = model.fittedvalues
resid = model.resid

# 1. Residuals vs Fitted
axes[0, 0].scatter(fitted / 1000, resid / 1000, alpha=0.4, color='steelblue', s=20)
axes[0, 0].axhline(0, color='red', linewidth=1.5)
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted Values ($000s)')
axes[0, 0].set_ylabel('Residuals ($000s)')

# 2. Q-Q plot
(osm, osr), (slope, intercept, _) = stats.probplot(resid)
axes[0, 1].scatter(osm, osr / 1000, alpha=0.4, color='steelblue', s=20)
axes[0, 1].plot([osm.min(), osm.max()],
                [(slope * osm.min() + intercept) / 1000,
                 (slope * osm.max() + intercept) / 1000], 'r-', linewidth=2)
axes[0, 1].set_title('Q-Q Plot of Residuals')
axes[0, 1].set_xlabel('Theoretical Quantiles')
axes[0, 1].set_ylabel('Sample Quantiles ($000s)')

# 3. Histogram of residuals
axes[1, 0].hist(resid / 1000, bins=30, color='steelblue', edgecolor='white')
axes[1, 0].set_title('Histogram of Residuals')
axes[1, 0].set_xlabel('Residuals ($000s)')
axes[1, 0].set_ylabel('Count')

# 4. Scale-Location
axes[1, 1].scatter(fitted / 1000, np.sqrt(np.abs(resid / 1000)),
                   alpha=0.4, color='steelblue', s=20)
axes[1, 1].set_title('Scale-Location Plot')
axes[1, 1].set_xlabel('Fitted Values ($000s)')
axes[1, 1].set_ylabel('√|Residuals|')

plt.tight_layout()
plt.show()

# **Interpretation of diagnostics:**
# - Residuals vs Fitted: Fan-shaped pattern indicates heteroscedasticity (variance increases
#   with salary level). Standard errors may be underestimated at high salary ranges.
# - Q-Q Plot: Residuals deviate from normality at the tails (right-skewed, heavy tails).
#   Jarque-Bera test: p < 0.001, kurtosis = 10.7.
# - Scale-Location: Confirms heteroscedasticity — spread grows with fitted values.
# - Durbin-Watson = 1.815: No significant autocorrelation (cross-sectional data).
#
# **Recommendations to improve model:**
# 1. Log-transform Salary to address heteroscedasticity and non-normality.
# 2. Investigate the $350 salary outlier — likely a data entry error.
# 3. Apply robust standard errors (HC3) to get valid inference despite heteroscedasticity.
# 4. Explore interaction terms (e.g., experience × education level).


# ============================================================
# STANDOUT: Regression Assumptions Deep Dive
# ============================================================

# In[23]:

from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF for multicollinearity check
X_vif = X_reduced.drop('const', axis=1)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i)
                   for i in range(X_vif.shape[1])]
print("Variance Inflation Factors:")
print(vif_data.sort_values('VIF', ascending=False))
# All VIF < 5 → no meaningful multicollinearity concern

# In[24]:

# Cook's Distance — identify high-influence points
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
n = len(y)
threshold = 4 / n  # common rule of thumb

high_influence = np.where(cooks_d > threshold)[0]
print(f"\nHigh-influence points (Cook's D > {threshold:.4f}): {len(high_influence)}")
print(f"Indices: {high_influence[:10]}...")  # show first 10

plt.figure(figsize=(10, 4))
plt.stem(range(n), cooks_d, markerfmt=',', linefmt='C0-', basefmt='k-')
plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold (4/n = {threshold:.3f})')
plt.title("Cook's Distance")
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.legend()
plt.tight_layout()
plt.show()
