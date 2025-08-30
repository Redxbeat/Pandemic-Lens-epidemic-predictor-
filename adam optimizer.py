from tensorflow.keras.optimizers import SGD

# Define SGD optimizer with momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile the model with SGD
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])