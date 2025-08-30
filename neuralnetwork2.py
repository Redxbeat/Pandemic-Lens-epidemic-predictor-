import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
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

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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
    Dense(2, activation='linear')  # Linear activation for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define the learning rate scheduler function
def lr_scheduler(epoch, lr, initial_lr=0.001, reduction_factor=0.95, start_epoch=50, stop_epoch=100):
    if epoch < start_epoch:
        return initial_lr  # Keep initial learning rate
    elif epoch < stop_epoch:
        return lr * reduction_factor  # Reduce learning rate by the reduction factor
    else:
        return lr  # Keep the learning rate constant after stop_epoch

# Define the LearningRateScheduler callback with additional parameters
lr_callback = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch, lr: lr_scheduler(epoch, lr, initial_lr=0.001, reduction_factor=0.95, start_epoch=50, stop_epoch=100)
)

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=[lr_callback])

print("Learning rate scheduler callback defined successfully.")
