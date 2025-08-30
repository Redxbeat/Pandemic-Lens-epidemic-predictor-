from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

# Load dataset
df = pd.read_csv('Final_epidemic_data.csv')

# Drop non-numeric columns and target columns
X = df.drop(columns=['Infection Rate', 'Mortality Rate', 'ISO Code', 'Country', 'Disease'])

# Define targets (y) - predicting both Infection Rate & Mortality Rate
y_infection = df['Infection Rate']
y_mortality = df['Mortality Rate']

# Convert non-numeric columns to numeric (if any)
X = X.apply(pd.to_numeric, errors='coerce')

# Fill any remaining NaNs with the mean of the column
X.fillna(X.mean(), inplace=True)

# Select top 10 features for Infection Rate
selector_infection = SelectKBest(score_func=f_regression, k=10)
X_new_infection = selector_infection.fit_transform(X, y_infection)

# Select top 10 features for Mortality Rate
selector_mortality = SelectKBest(score_func=f_regression, k=10)
X_new_mortality = selector_mortality.fit_transform(X, y_mortality)

print("Top 10 features selected successfully for Infection Rate and Mortality Rate.")