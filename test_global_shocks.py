from simulator import simulate_shock
import torch

def test_global_shocks():
    # Test case 1: Truckers Strike (Strikes: 1.10)
    print("Testing 'Truckers Strike'...")
    res1 = simulate_shock("Truckers strike on Delhi-Mumbai highway", commodity="ONION")
    print(f"Origin: {res1['origin_name']}")
    print(f"Shock Type: {res1['features']['shock_type']}")
    print(f"Multiplier: {res1['features']['impact_multiplier']}")
    assert res1['features']['shock_type'] == 'global'
    assert res1['features']['impact_multiplier'] == 1.10
    
    # Test case 2: Drought (Weather: 1.15)
    print("\nTesting 'Drought'...")
    res2 = simulate_shock("Severe drought affecting crop yields", commodity="ONION")
    print(f"Origin: {res2['origin_name']}")
    print(f"Shock Type: {res2['features']['shock_type']}")
    print(f"Multiplier: {res2['features']['impact_multiplier']}")
    assert res2['features']['shock_type'] == 'global'
    assert res2['features']['impact_multiplier'] == 1.15

    # Test case 3: Policy Change (Policy: 1.20)
    print("\nTesting 'Policy Change'...")
    res3 = simulate_shock("Government announces new policy change for export duties", commodity="ONION")
    print(f"Origin: {res3['origin_name']}")
    print(f"Shock Type: {res3['features']['shock_type']}")
    print(f"Multiplier: {res3['features']['impact_multiplier']}")
    assert res3['features']['shock_type'] == 'global'
    assert res3['features']['impact_multiplier'] == 1.20

    # Test case 4: Single Mandi (LLM Fallback)
    print("\nTesting Single Mandi (Nashik)...")
    res4 = simulate_shock("Heavy rain in Nashik market", commodity="ONION")
    print(f"Origin: {res4['origin_name']}")
    print(f"Shock Type: {res4['features']['shock_type']}")
    # This should use LLM or Mock (which might return climatic/1.3 for "rain")
    assert res4['origin_name'] != 'GLOBAL (Mean Impact)'

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_global_shocks()
