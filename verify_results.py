import pandas as pd

# Load the result created by the geocoder
df = pd.read_csv("market_coords.csv")

# Count how many markets now have coordinates
total = len(df)
with_coords = df['latitude'].notnull().sum()
missing = df['latitude'].isnull().sum()

print(f"📊 Final Geospatial Report:")
print(f"✅ Markets with Coordinates: {with_coords}")
print(f"❌ Markets still missing: {missing}")

# Show a sample of the districts that were successfully fixed
print("\nSample of Mapped Mandis:")
print(df[['District', 'latitude', 'longitude']].head(10))