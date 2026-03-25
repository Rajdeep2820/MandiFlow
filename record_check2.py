import requests

API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
NEW_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

# Let's test two different historical dates
test_dates = ["01/03/2026", "10/03/2026"]

for test_date in test_dates:
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1,
        "filters[arrival_date]": test_date
    }

    print(f"Pinging new API for {test_date}...")
    response = requests.get(NEW_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        total_records = data.get('total', 0)
        print(f"✅ SUCCESS: Found {total_records} records.")
        
        # Extra verification: Let's check the actual date of the first record returned!
        if total_records > 0 and 'records' in data and len(data['records']) > 0:
            actual_date = data['records'][0].get('arrival_date', 'Unknown')
            print(f"   -> Verification: The first record's actual date is {actual_date}")
    else:
        print(f"❌ FAILED: API returned status code {response.status_code}")
    print("-" * 40)