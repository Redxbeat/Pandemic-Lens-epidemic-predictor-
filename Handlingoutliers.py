import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the DataFrame from the CSV file
df = pd.read_csv('update_epidemic_data_filled_custom.csv')

# Visualize outliers in numerical features
for col in ["Population Density", "Vaccination Rate (%)", "Average Temperature (°C)", "Humidity (%)", "Mortality Rate", "Infection Rate", "Mobility Index"]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()