import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input layer
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),  # Increased dropout rate to prevent overfitting
    Dense(128, activation='relu'),  # Hidden layer
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),  # Hidden layer
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),  # Hidden layer
    BatchNormalization(),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='linear')  # Output layer (for regression)
])

# Compile the model with Adam optimizer
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Summary of the model
model.summary()

# Define early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, lr_scheduler])

print("Model trained successfully.")
# Save the trained model in the recommended Keras format
model.save('trained_model.keras')
print("Model saved to 'trained_model.keras'.")