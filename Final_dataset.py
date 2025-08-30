import pandas as pd

# Define the absolute path to the CSV file
file_path = r'B:\Python programming\MINI project\updated_epidemic_data_with_climate_index.csv'

# Load the DataFrame from the CSV file
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Save the DataFrame to a new CSV file
df.to_csv("Final_epidemic_data.csv", index=False)

print("Final dataset saved as Final_epidemic_data.csv")