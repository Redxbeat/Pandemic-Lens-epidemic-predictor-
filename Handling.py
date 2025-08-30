import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the DataFrame from the CSV file
df = pd.read_csv('update_epidemic_data_filled_custom.csv')

# Fill missing numerical values with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Verify missing values are handled
print("Missing values after filling:\n", df.isnull().sum())

# Select Features (X) and Target (y)
X = df[[
    'Population', 
    'Population Density', 
    'GDP per Capita', 
    'Healthcare Expenditure (% of GDP)', 
    'Hospital Beds per 1,000 People', 
    'Life Expectancy', 
    'Vaccination Rate (%)', 
    'Average Temperature (°C)', 
    'Humidity (%)', 
    'Mobility Index'
]]
y = df[['Infection Rate', 'Mortality Rate']]

# Reset the index of the target variable (y)
y = y.reset_index(drop=True)

# Split the data into training and validation/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Features and target variable selected, missing values handled, and indices reset.")