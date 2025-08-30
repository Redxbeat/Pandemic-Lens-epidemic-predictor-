from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('trained_model.keras')  # Ensure the model file is in the same directory or provide the correct path

# Load dataset
df = pd.read_csv('Final_epidemic_data.csv')

# Print the column names to verify
print("Columns in the dataset:", df.columns)

# Preprocess the data
# Ensure the same features are used as during training
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

# Handle missing values in X
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)

# Handle missing values in y
y = y.apply(pd.to_numeric, errors='coerce')
y.fillna(y.mean(), inplace=True)

# Reset the index of y to ensure proper alignment
y = y.reset_index(drop=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and validation sets
_, X_val, _, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verify the shape of X_val matches the input shape expected by the model
print(f"Shape of X_val: {X_val.shape}")
print(f"Expected input shape: {model.input_shape}")

# Make predictions on the validation set
y_pred = model.predict(X_val)  # Pass X_val, not y_val

# Convert predictions to a DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=["Predicted Infection Rate", "Predicted Mortality Rate"])
y_true_df = pd.DataFrame(y_val, columns=["Actual Infection Rate", "Actual Mortality Rate"])

# Display comparison
comparison = pd.concat([y_true_df.reset_index(drop=True), y_pred_df], axis=1)
print(comparison.head(12))  # Show first 12 comparisons