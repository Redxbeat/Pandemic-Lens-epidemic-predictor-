import pandas as pd

# Load the processed dataset from Feature Engineering step
df = pd.read_csv("update_epidemic_data_swapped.csv")

# Display the first few rows
print(df.head())