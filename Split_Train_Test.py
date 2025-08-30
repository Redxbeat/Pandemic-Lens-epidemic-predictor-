import pandas as pd
from sklearn.model_selection import train_test_split

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

# Fill NaN values in the target variable (y) with the column mean
y.fillna(y.mean(), inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reset the index of y_test to ensure proper alignment
y_test = y_test.reset_index(drop=True)

# Example: Concatenate y_test with predictions (assuming predictions are available)
# For demonstration, let's create a dummy predictions DataFrame
import numpy as np
y_pred = pd.DataFrame(np.random.rand(len(y_test), 2), columns=["Predicted Infection Rate", "Predicted Mortality Rate"])

# Concatenate actual and predicted values
comparison = pd.concat([y_test, y_pred], axis=1)

# Display the first 10 rows of the comparison DataFrame
print(comparison.head(10))