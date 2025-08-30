import pandas as pd

# Load the processed dataset from Feature Engineering step
df = pd.read_csv("Final_epidemic_data.csv")

# Display the first few rows
print(df.head())