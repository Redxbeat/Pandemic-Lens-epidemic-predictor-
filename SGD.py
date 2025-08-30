from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
df = pd.read_csv('Final_epidemic_data.csv')

# Drop non-numeric columns and target columns
X = df.drop(columns=['Infection Rate', 'Mortality Rate', 'ISO Code', 'Country', 'Disease'])

# Define targets (y) - predicting both Infection Rate & Mortality Rate
y = df[['Infection Rate', 'Mortality Rate']]

# Convert non-numeric columns to numeric (if any)
X = X.apply(pd.to_numeric, errors='coerce')

# Fill any remaining NaNs with the mean of the column
X.fillna(X.mean(), inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the model architecture
model = Sequential([
    Input(shape=(X_scaled.shape[1],)),  # Input layer
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])

# Define SGD optimizer with momentum
optimizer = SGD(learning_rate=0.01, momentum=0.95, nesterov=True)

# Compile the model with SGD
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("Model compiled successfully with SGD optimizer.")