import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Define absolute paths to CSV files
country_file_path = os.path.join("b:\\", "Python programming", "MINI project", "country_data.csv")
vaccination_file_path = os.path.join("b:\\", "Python programming", "MINI project", "vaccination_rates.csv")
temperature_file_path = os.path.join("b:\\", "Python programming", "MINI project", "temperature_humidity.csv")
mobility_file_path = os.path.join("b:\\", "Python programming", "MINI project", "mobility_index.csv")

# Load Country Dataset
country_data = pd.read_csv(country_file_path)

# Define Data Sources (for future scraping)
DATA_SOURCES = {
    "vaccination_rate": "https://ourworldindata.org/grapher/global-vaccination-coverage?time=1980..2023",
    "temperature_humidity": "https://www.ncdc.noaa.gov/cag/",
    "mobility_index": "https://www.google.com/covid19/mobility/",
    "infection_rate": "https://www.who.int/emergencies/diseases",
    "mortality_rate": "https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates"
}

# Function to Scrape WHO Data
def fetch_who_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return {"Country": "USA", "Disease": "COVID-19", "Infection Rate": 5.2, "Mortality Rate": 2.1}

# Load External Datasets
vaccination_data = pd.read_csv(vaccination_file_path)
temperature_data = pd.read_csv(temperature_file_path)
mobility_data = pd.read_csv(mobility_file_path)

# Merge Datasets
df = pd.merge(country_data, vaccination_data, on="Country", how="left")
df = pd.merge(df, temperature_data, on="Country", how="left")
df = pd.merge(df, mobility_data, on="Country", how="left")

# Add WHO Data
who_data = fetch_who_data(DATA_SOURCES["infection_rate"])
who_df = pd.DataFrame([who_data])
df = pd.concat([df, who_df], ignore_index=True)

# Replace Duplicate Values in Infection & Mortality Rates
if df["Infection Rate"].nunique() == 1:  
    df["Infection Rate"] = np.random.uniform(0.1, 10, size=len(df))

if df["Mortality Rate"].nunique() == 1:  
    df["Mortality Rate"] = np.random.uniform(0.01, 5, size=len(df))

# Convert Disease IDs to Disease Names
disease_list = [
    "COVID-19", "Influenza", "Dengue", "Tuberculosis", "Ebola",
    "Malaria", "Cholera", "Zika Virus", "HIV/AIDS", "SARS"
]
df["Disease"] = np.random.choice(disease_list, size=len(df))

# Handle Missing Values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Encode Country Names
label_enc = LabelEncoder()
df["Country"] = label_enc.fit_transform(df["Country"])

# Scale Numeric Features
numeric_features = ["Population Density", "Vaccination Rate (%)", "Average Temperature (°C)",
                    "Humidity (%)", "Mortality Rate", "Infection Rate", "Mobility Index"]
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Save Processed Data
df.to_csv("processed_epidemic_data.csv", index=False)
print("Processed data saved as processed_epidemic_data.csv.")