import requests

API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
URL_9ef8 = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
URL_3598 = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

def test_api(url, date_str):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1,
        "filters[arrival_date]": date_str
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        records = data.get("records", [])
        if records:
            actual_date = records[0].get("arrival_date", "Missing")
            total = data.get("total", "Unknown")
            print(f"[{url.split('/')[-1]}] req={date_str} -> returned={actual_date} | total={total}")
        else:
            print(f"[{url.split('/')[-1]}] req={date_str} -> NO RECORDS | total={data.get('total')}")
    else:
        print(f"[{url.split('/')[-1]}] Error {resp.status_code}")

print("Testing Current Resource (9ef8...):")
test_api(URL_9ef8, "01/01/2026")
test_api(URL_9ef8, "21/03/2026")

print("\nTesting Alternative Resource (3598...):")
test_api(URL_3598, "01/01/2026")
test_api(URL_3598, "21/03/2026")
