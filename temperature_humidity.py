import pandas as pd
import numpy as np

# Use the same list of 100 countries
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

# Generate random temperature (°C) and humidity (%) values
np.random.seed(42)
avg_temperatures = np.random.uniform(-10, 40, len(countries))  # Between -10°C (cold) and 40°C (hot)
humidity_levels = np.random.uniform(20, 90, len(countries))  # Humidity percentage between 20% and 90%

# Create DataFrame
df_temp_humidity = pd.DataFrame({
    "Country": countries,
    "Average Temperature (°C)": np.round(avg_temperatures, 1),  # Round to 1 decimal place
    "Humidity (%)": np.round(humidity_levels, 1)  # Round to 1 decimal place
})

# Save to CSV
df_temp_humidity.to_csv("temperature_humidity.csv", index=False)

print("Temperature and humidity dataset 'temperature_humidity.csv' generated successfully!")