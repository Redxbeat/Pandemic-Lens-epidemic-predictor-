from tensorflow.keras.models import load_model
import pandas as pd

# Load the trained model
model = load_model('trained_model.keras')  # Ensure the model file is in the same directory or provide the correct path

# Predict Infection & Mortality Rates
y_pred = model.predict(X_val)

# Convert predictions to DataFrame for better visualization
y_pred_df = pd.DataFrame(y_pred, columns=["Predicted Infection Rate", "Predicted Mortality Rate"])
y_true_df = pd.DataFrame(y_val, columns=["Actual Infection Rate", "Actual Mortality Rate"])

# Compare actual vs predicted values
comparison = pd.concat([y_true_df.reset_index(drop=True), y_pred_df], axis=1)
print(comparison.head(10))  # Print first 10 comparisons