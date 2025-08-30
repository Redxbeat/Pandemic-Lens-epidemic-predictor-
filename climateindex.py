import pandas as pd

# Load the DataFrame from the CSV file
df = pd.read_csv('updated_epidemic_data.csv')

# Calculate the Climate Index
df["Climate Index"] = (df["Average Temperature (°C)"] + df["Humidity (%)"]) / 2

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_epidemic_data_with_climate_index.csv', index=False)

print("Climate Index calculated and data saved to 'update_epidemic_data_with_climate_index.csv'.")