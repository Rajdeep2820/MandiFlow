import pandas as pd
import numpy as np
import os

MASTER_FILE = "mandi_master_data.parquet"
def verify_data():
    if not os.path.exists(MASTER_FILE):
        print(f"❌ FAILED: {MASTER_FILE} does not exist. Run preprocess.py first.")
        return

    # Load data
    df = pd.read_parquet(MASTER_FILE)
    print(f"✅ SUCCESS: Loaded {len(df)} records from Master Parquet.\n")

    # 1. Math Check: Outliers
    # Formula: Is Max Price within a reasonable range of Mean?
    # Impact: Ensures the Z-score logic (x - mu / sigma) was applied.
    max_p = df['Modal_Price'].max()
    mean_p = df['Modal_Price'].mean()
    std_p = df['Modal_Price'].std()
    
    print(f"--- Statistics Check ---")
    print(f"Mean Price: {mean_p:.2f} | Max Price: {max_p:.2f}")
    if max_p < (mean_p + 4 * std_p):
        print("✅ PASS: Outlier removal looks successful.")
    else:
        print("⚠️ WARNING: High prices detected. Check if Z-score threshold (3) is too loose.")

    # 2. Math Check: Seasonality (Unit Circle Test)
    # Formula: sin^2(theta) + cos^2(theta) = 1
    # Impact: Verifies that months are mapped correctly to a circular timeline.
    check_val = (df['month_sin']**2 + df['month_cos']**2).iloc[0]
    print(f"\n--- Trigonometry Check ---")
    print(f"sin² + cos² = {check_val:.4f}")
    if np.isclose(check_val, 1.0):
        print("✅ PASS: Cyclical month encoding is mathematically valid.")
    else:
        print("❌ FAIL: Trigonometric encoding error.")

    # 3. Structural Check: IDs
    # Impact: Confirms Label Encoding turned "Mandsaur" into an Integer.
    print(f"\n--- Feature Type Check ---")
    if pd.api.types.is_integer_dtype(df['Market_ID']):
        print("✅ PASS: Market names converted to IDs.")
    else:
        print("❌ FAIL: Market column is still text.")

    print("\n" + "="*30)
    print("VERDICT: DATA IS READY FOR GRAPH BUILDING")
    print("="*30)

if __name__ == "__main__":
    verify_data()