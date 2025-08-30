import pandas as pd
import numpy as np

# Load the DataFrame from the CSV file
df = pd.read_csv('update_epidemic_data_filled_custom.csv')

# Calculate the first and third quartiles (Q1 and Q3) for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Remove rows where data is outside 1.5*IQR range for numeric columns
df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# For non-numeric columns, handle outliers based on frequency
categorical_cols = df.select_dtypes(include=[object]).columns
for col in categorical_cols:
    freq = df[col].value_counts()
    rare_labels = freq[freq < 2].index  # Define a threshold for rare labels
    df = df[~df[col].isin(rare_labels)]

# Save the cleaned DataFrame to a new CSV file
df.to_csv('clean_epidemic_data.csv', index=False)

print("Outliers removed and data saved to 'clean_epidemic_data.csv'.")