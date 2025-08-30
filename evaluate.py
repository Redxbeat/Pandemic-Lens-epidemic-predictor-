import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv('Final_epidemic_data.csv')

# Drop non-numeric columns and target columns
X = df.drop(columns=['Infection Rate', 'Mortality Rate', 'ISO Code', 'Country', 'Disease'])
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
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='linear')  # Output layer
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))

# Save the history object to a file
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Training history saved to 'history.pkl'.")

# Load the history object
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plot training and validation loss
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation MAE
plt.plot(history['mae'], label='Training MAE')
plt.plot(history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

