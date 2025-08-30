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

# Generate random mobility index values (e.g., between 0 and 100)
np.random.seed(42)
mobility_index = np.random.uniform(0, 100, len(countries))

# Create DataFrame
df_mobility = pd.DataFrame({
    "Country": countries,
    "Mobility Index": np.round(mobility_index, 2)  # Round to 2 decimal places
})

# Save to CSV
df_mobility.to_csv("mobility_index.csv", index=False)

print("Mobility index dataset 'mobility_index.csv' generated successfully!")