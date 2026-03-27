import os

if os.path.exists("mandi_adjacency_index_onion.txt"):
    with open("mandi_adjacency_index_onion.txt", "r") as f:
        markets = [line.strip().upper() for line in f.readlines()]
        
    print(f"🧠 Your trained model knows {len(markets)} markets.")
    print("Here are the first 15 markets you can test right now:")
    print(markets[:15])
    
    # Let's search for Nashik variants
    nashik_variants = [m for m in markets if 'NAS' in m or 'LASAL' in m]
    print(f"\nPotential matches for Nashik/Lasalgaon: {nashik_variants}")
else:
    print("Index file not found!")