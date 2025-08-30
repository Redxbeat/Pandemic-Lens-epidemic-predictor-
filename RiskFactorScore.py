import pandas as pd

# Load the DataFrame from the CSV file
df = pd.read_csv('update_epidemic_data_filled_custom.csv')

# Calculate the Risk Factor
df["Risk Factor"] = df["Infection Rate"] * df["Mortality Rate"]

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_epidemic_data.csv', index=False)

print("Risk Factor calculated and data saved to 'updated_epidemic_data.csv'.")