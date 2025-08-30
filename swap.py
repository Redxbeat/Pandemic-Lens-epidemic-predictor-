import pandas as pd

# Load the DataFrame from the CSV file
df = pd.read_csv('processed_epidemic_data.csv')

# Swap the values in the "Country" and "ISO Code" columns
df['Country'], df['ISO Code'] = df['ISO Code'], df['Country']

# Verify the swap
print(df[['Country', 'ISO Code']].head())

# Save the updated dataset to a new CSV file
df.to_csv("update_epidemic_data_swapped.csv", index=False)