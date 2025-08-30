import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Load the DataFrame from the CSV file
df = pd.read_csv('update_epidemic_data_filled_custom.csv')

# Compute correlation matrix for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_corr_matrix = df[numeric_cols].corr()

# Function to calculate Cramér's V for categorical columns
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

# Compute correlation matrix for non-numeric columns
categorical_cols = df.select_dtypes(include=[object]).columns
categorical_corr_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

for col1 in categorical_cols:
    for col2 in categorical_cols:
        if col1 == col2:
            categorical_corr_matrix.loc[col1, col2] = 1.0
        else:
            categorical_corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

# Convert the categorical correlation matrix to numeric
categorical_corr_matrix = categorical_corr_matrix.astype(float)

# Visualize numeric correlation matrix
plt.figure(figsize=(12, 6))
sns.heatmap(numeric_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numeric Feature Correlation Matrix")
plt.show()

# Visualize categorical correlation matrix
plt.figure(figsize=(12, 6))
sns.heatmap(categorical_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Categorical Feature Correlation Matrix")
plt.show()