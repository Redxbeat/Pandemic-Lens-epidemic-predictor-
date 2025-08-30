import pandas as pd

# Load the DataFrame from the CSV file
df = pd.read_csv('Final_epidemic_data.csv')

# Print the column names to verify
print("Columns in Final_epidemic_data.csv:", df.columns)

# Select Features (X) and Target (y)
X = df[[
    'Population', 
    'Population Density', 
    'GDP per Capita', 
    'Healthcare Expenditure (% of GDP)',  # Adjusted column name
    'Hospital Beds per 1,000 People',     # Adjusted column name
    'Life Expectancy', 
    'Vaccination Rate (%)', 
    'Average Temperature (°C)', 
    'Humidity (%)', 
    'Mobility Index'
]]

y = df[['Infection Rate', 'Mortality Rate']]
print("Features and target variable selected.")