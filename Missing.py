import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the absolute path to the CSV file
file_path = r'b:\Python programming\MINI project\update_epidemic_data_swapped.csv'
country_file_path = r'b:\Python programming\MINI project\country_data.csv'

# Load the DataFrame from the CSV file
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Load the country data for mapping ISO codes to country names
country_data = pd.read_csv(country_file_path)

# Print the column names to verify
print("Columns in country_data.csv:", country_data.columns)

# Assuming country_data.csv has columns 'ISO Code' and 'Country'
# Adjust these column names based on the actual column names in your country_data.csv file
iso_to_country = dict(zip(country_data['ISO Code'], country_data['Country']))

# Fill missing values in the "Country" column using the ISO code mapping
df['Country'] = df.apply(lambda row: iso_to_country.get(row['ISO Code'], row['Country']), axis=1)

# Display missing values count after filling
print("Missing values per column after filling:\n", df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='coolwarm', cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()