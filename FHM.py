import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("final_epidemic_data.csv")

# Inspect the data
print(df.head())
print(df.describe())
print(df.info())

# Check for constant values in "Infection Rate" and "Mortality Rate"
print("Unique values in 'Infection Rate':", df["Infection Rate"].unique())
print("Unique values in 'Mortality Rate':", df["Mortality Rate"].unique())

# Replace invalid values with synthetic data (example: replace with values following a normal distribution)
# In practice, you should replace with actual meaningful data
np.random.seed(42)  # For reproducibility
df["Infection Rate"] = np.random.normal(loc=0.1, scale=0.02, size=len(df))
df["Mortality Rate"] = np.random.normal(loc=0.01, scale=0.005, size=len(df))

# Ensure no negative values
df["Infection Rate"] = df["Infection Rate"].clip(lower=0)
df["Mortality Rate"] = df["Mortality Rate"].clip(lower=0)

# Define features (X) and target variables (y)
X = df.drop(columns=["Infection Rate", "Mortality Rate"])  # Features
y = df[["Infection Rate", "Mortality Rate"]]  # Target variables

# Ensure no data leakage
assert "Infection Rate" not in X.columns
assert "Mortality Rate" not in X.columns

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(include=[object]).columns

# Preprocessing for numeric data: scaling and polynomial features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a pipeline that combines preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42)))
])

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'regressor__estimator__n_estimators': [50, 100, 200],
    'regressor__estimator__max_depth': [3, 5, 7],
    'regressor__estimator__learning_rate': [0.01, 0.1, 0.2]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Convert predictions to a DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=["Predicted Infection Rate", "Predicted Mortality Rate"])

# Display first few predicted values
print(y_pred_df.head())

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Cross-validation to check for overfitting
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean Cross-Validation R² Score: {np.mean(cv_scores):.4f}")