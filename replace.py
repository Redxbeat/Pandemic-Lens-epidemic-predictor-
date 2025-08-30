import pandas as pd

# Load the DataFrame from the CSV file
df = pd.read_csv('processed_epidemic_data.csv')

# Load the country data for mapping ISO codes to country names
country_data = pd.read_csv('country_data.csv')

# Print the column names to verify
print("Columns in country_data.csv:", country_data.columns)

# Assuming country_data.csv has columns 'ISO_Code' and 'Country_Name'
# Adjust these column names based on the actual column names in your country_data.csv file
iso_to_country = dict(zip(country_data['ISO_Code'], country_data['Country_Name']))

# Replace ISO codes with country names in the 'Country' column
df['Country'] = df['ISO Code'].map(iso_to_country)

# Verify if missing values are filled
print("Missing values after filling:\n", df.isnull().sum())

# Display updated DataFrame
print(df[['Country', 'ISO Code']].head())

# Save the updated dataset to a new CSV file
df.to_csv("updated_epidemic_data.csv", index=False)