from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop

# Example of using Dense with L2 regularization
dense_layer = Dense(256, activation='relu', kernel_regularizer=l2(0.01))

# Example of defining an RMSprop optimizer
optimizer = RMSprop(learning_rate=0.001)

print("Dense layer and optimizer defined successfully.")