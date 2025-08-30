import pandas as pd
import numpy as np

# List of 100 countries with their ISO codes
countries = [
    ("United States", "US"), ("India", "IN"), ("Brazil", "BR"), ("Germany", "DE"),
    ("United Kingdom", "GB"), ("China", "CN"), ("Russia", "RU"), ("South Africa", "ZA"),
    ("Australia", "AU"), ("Canada", "CA"), ("France", "FR"), ("Italy", "IT"),
    ("Spain", "ES"), ("Mexico", "MX"), ("Japan", "JP"), ("South Korea", "KR"),
    ("Indonesia", "ID"), ("Saudi Arabia", "SA"), ("Argentina", "AR"), ("Turkey", "TR"),
    ("Netherlands", "NL"), ("Sweden", "SE"), ("Switzerland", "CH"), ("Thailand", "TH"),
    ("Poland", "PL"), ("Belgium", "BE"), ("Nigeria", "NG"), ("Egypt", "EG"),
    ("Vietnam", "VN"), ("Philippines", "PH"), ("Pakistan", "PK"), ("Bangladesh", "BD"),
    ("Colombia", "CO"), ("Malaysia", "MY"), ("Chile", "CL"), ("United Arab Emirates", "AE"),
    ("Ukraine", "UA"), ("Czech Republic", "CZ"), ("Portugal", "PT"), ("Greece", "GR"),
    ("Romania", "RO"), ("Austria", "AT"), ("Norway", "NO"), ("Israel", "IL"),
    ("Singapore", "SG"), ("Denmark", "DK"), ("Hungary", "HU"), ("Ireland", "IE"),
    ("New Zealand", "NZ"), ("Finland", "FI"), ("Kazakhstan", "KZ"), ("Algeria", "DZ"),
    ("Peru", "PE"), ("Iraq", "IQ"), ("Morocco", "MA"), ("Ecuador", "EC"),
    ("Qatar", "QA"), ("Serbia", "RS"), ("Belarus", "BY"), ("Venezuela", "VE"),
    ("Sri Lanka", "LK"), ("Uzbekistan", "UZ"), ("Dominican Republic", "DO"),
    ("Kenya", "KE"), ("Sudan", "SD"), ("Bolivia", "BO"), ("Bulgaria", "BG"),
    ("Croatia", "HR"), ("Slovakia", "SK"), ("Tunisia", "TN"), ("Lebanon", "LB"),
    ("Jordan", "JO"), ("Costa Rica", "CR"), ("Lithuania", "LT"), ("Oman", "OM"),
    ("Paraguay", "PY"), ("Latvia", "LV"), ("Estonia", "EE"), ("Bahrain", "BH"),
    ("Cyprus", "CY"), ("Iceland", "IS"), ("Mongolia", "MN"), ("Panama", "PA"),
    ("Kuwait", "KW"), ("Georgia", "GE"), ("Uruguay", "UY"), ("Jamaica", "JM"),
    ("Armenia", "AM"), ("Albania", "AL"), ("Botswana", "BW"), ("Malta", "MT"),
    ("Brunei", "BN"), ("Namibia", "NA"), ("Ghana", "GH"), ("Ethiopia", "ET"),
    ("Zambia", "ZM"), ("Senegal", "SN"), ("Guatemala", "GT"), ("Nepal", "NP"),
    ("Honduras", "HN"), ("Bosnia and Herzegovina", "BA"), ("Madagascar", "MG"),
    ("Zimbabwe", "ZW"), ("Malawi", "MW"), ("Rwanda", "RW")
]

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data = {
    "Country": [country[0] for country in countries],
    "ISO Code": [country[1] for country in countries],
    "Population": np.random.randint(500_000, 1_500_000_000, len(countries)),  # Population range
    "Population Density": np.random.uniform(5, 1500, len(countries)),  # People per km²
    "GDP per Capita": np.random.uniform(500, 80000, len(countries)),  # GDP per capita in USD
    "Healthcare Expenditure (% of GDP)": np.random.uniform(2, 15, len(countries)),  # Healthcare spending
    "Hospital Beds per 1,000 People": np.random.uniform(0.5, 10, len(countries)),  # Hospital beds per 1000 people
    "Life Expectancy": np.random.uniform(50, 85, len(countries))  # Life expectancy in years
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("country_data.csv", index=False)

print("Country dataset 'country_data.csv' with 100 countries generated successfully!")