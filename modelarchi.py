from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
df = pd.read_csv('Final_epidemic_data.csv')

# Drop non-numeric columns and target columns
X = df.drop(columns=['Infection Rate', 'Mortality Rate', 'ISO Code', 'Country', 'Disease'])

# Convert non-numeric columns to numeric (if any)
X = X.apply(pd.to_numeric, errors='coerce')

# Fill any remaining NaNs with the mean of the column
X.fillna(X.mean(), inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the model architecture
model = Sequential()
model.add(Input(shape=(X_scaled.shape[1],)))  # Input layer
model.add(Dense(128, activation='relu'))  # More neurons
model.add(Dropout(0.3))  # Prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Predicting continuous values (infection/mortality rate)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Model architecture defined successfully.")