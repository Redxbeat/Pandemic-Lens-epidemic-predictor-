import pandas as pd

# Load the DataFrame from the CSV file
df = pd.read_csv('processed_epidemic_data.csv')

# Save the DataFrame to a new CSV file
df.to_csv("final_epidemic_data.csv", index=False)

print("Final dataset saved as final_epidemic_data.csv")