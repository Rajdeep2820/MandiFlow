import requests

API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

dates = ["21/03/2026", "22/03/2026", "23/03/2026", "24/03/2026", "25/03/2026"]

print("Fetching total actual records available for each day on gov server...\n")
for date in dates:
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1,
        "filters[arrival_date]": date
    }
    try:
        resp = requests.get(URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("total", "Unknown")
            print(f"Date: {date} -> Total Records Available: {total}")
        else:
            print(f"Date: {date} -> API Error {resp.status_code}")
    except Exception as e:
        print(f"Date: {date} -> Exception {e}")
