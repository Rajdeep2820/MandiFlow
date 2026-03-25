import requests

API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

params = {
    "api-key": API_KEY,
    "format": "json",
    "limit": 1,
    "filters[arrival_date]": "21/03/2026"
}

print("Pinging Agmarknet API for 21th March...")
response = requests.get(URL, params=params)

if response.status_code == 200:
    data = response.json()
    print(f"✅ SUCCESS: The API has exactly {data.get('total', 0)} records for 21/03/2026.")
else:
    print(f"❌ FAILED: API returned status code {response.status_code}")