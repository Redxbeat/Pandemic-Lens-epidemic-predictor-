import tensorflow as tf

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

print("Learning rate scheduler callback defined successfully.")