import pandas as pd
import numpy as np

# List of 100 countries (Same as used before)
countries = [
    "United States", "India", "Brazil", "Germany", "United Kingdom", "China", "Russia",
    "South Africa", "Australia", "Canada", "France", "Italy", "Spain", "Mexico", "Japan",
    "South Korea", "Indonesia", "Saudi Arabia", "Argentina", "Turkey", "Netherlands",
    "Sweden", "Switzerland", "Thailand", "Poland", "Belgium", "Nigeria", "Egypt", "Vietnam",
    "Philippines", "Pakistan", "Bangladesh", "Colombia", "Malaysia", "Chile",
    "United Arab Emirates", "Ukraine", "Czech Republic", "Portugal", "Greece", "Romania",
    "Austria", "Norway", "Israel", "Singapore", "Denmark", "Hungary", "Ireland", "New Zealand",
    "Finland", "Kazakhstan", "Algeria", "Peru", "Iraq", "Morocco", "Ecuador", "Qatar",
    "Serbia", "Belarus", "Venezuela", "Sri Lanka", "Uzbekistan", "Dominican Republic",
    "Kenya", "Sudan", "Bolivia", "Bulgaria", "Croatia", "Slovakia", "Tunisia", "Lebanon",
    "Jordan", "Costa Rica", "Lithuania", "Oman", "Paraguay", "Latvia", "Estonia", "Bahrain",
    "Cyprus", "Iceland", "Mongolia", "Panama", "Kuwait", "Georgia", "Uruguay", "Jamaica",
    "Armenia", "Albania", "Botswana", "Malta", "Brunei", "Namibia", "Ghana", "Ethiopia",
    "Zambia", "Senegal", "Guatemala", "Nepal", "Honduras", "Bosnia and Herzegovina",
    "Madagascar", "Zimbabwe", "Malawi", "Rwanda"
]

# Generate random vaccination rates (in percentage)
np.random.seed(42)
vaccination_rates = np.random.uniform(10, 95, len(countries))  # Vaccination rate between 10% to 95%

# Create DataFrame
df_vaccination = pd.DataFrame({
    "Country": countries,
    "Vaccination Rate (%)": np.round(vaccination_rates, 2)  # Round to 2 decimal places
})

# Save to CSV
df_vaccination.to_csv("vaccination_rates.csv", index=False)

print("Vaccination rate dataset 'vaccination_rates.csv' generated successfully!")