import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the absolute path to the CSV file
file_path = r'B:\Python programming\MINI project\updated_epidemic_data_with_climate_index.csv'

# Load the DataFrame from the CSV file
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Plot the histogram and KDE of Population Density
sns.histplot(df['Population Density'], kde=True, bins=30)
plt.title('Histogram & KDE of Population Density')
plt.show()