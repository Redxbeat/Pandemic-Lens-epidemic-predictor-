import pandas as pd

# Load the processed dataset
df = pd.read_csv("updated_epidemic_data_swapped.csv")

# Display the first few rows
print(df.head())

# Get dataset structure and summary
print(df.info())

# Get statistical summary of numerical features
print(df.describe())