import pandas as pd

# Load the dataset
df = pd.read_csv("processed_epidemic_data.csv")

# Check for duplicate rows
duplicates = df[df.duplicated()]

# Print the duplicated rows
print("Duplicated rows:")
print(duplicates)

# Check for duplicate values in specific columns
for column in df.columns:
    duplicate_values = df[df[column].duplicated()][column]
    if not duplicate_values.empty:
        print(f"Duplicated values in column '{column}':")
        print(duplicate_values)