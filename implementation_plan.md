# Fixing Geocoding State Collisions

The current [geocoder.py](file:///Users/rajdeepsinghpanwar/Downloads/MandiFlow/geocoder.py) matches district names purely across the entire India pincode dataset without considering state boundaries. This causes words like `"DHARASHIV"` (Maharashtra) to get fuzzy-assigned to `"DHAR"` (Madhya Pradesh).

## Proposed Changes

### [geocoder.py](file:///Users/rajdeepsinghpanwar/Downloads/MandiFlow/geocoder.py)
We will rewrite the district matching algorithm to completely eradicate cross-border misplacement.

#### [MODIFY] [geocoder.py](file:///Users/rajdeepsinghpanwar/Downloads/MandiFlow/geocoder.py)
- Update `master_df` loading to also query the `State` column.
- Update `pincode_df` loading to retain the `statename` column.
- Refactor [find_best_match](file:///Users/rajdeepsinghpanwar/Downloads/MandiFlow/geocoder.py#48-68) to first fuzzy-match the **State** against a known state list, and then *exclusively* fuzzy match the **District** against the allowed bounding box of that state.
- Immediately add known modern renamed district translations into the `manual_map`:
    - `"DHARASHIV"` ➡️ `"OSMANABAD"`
    - `"CHHATRAPATI SAMBHAJINAGAR"` ➡️ `"AURANGABAD"`

## Verification Plan

### Automated Verification
After rewriting the wrapper, we will manually execute `python geocoder.py` in the terminal to trace its performance. If any state name fails to map, we'll implement a temporary fallback to the national dictionary.
Finally we read the newly generated [market_coords.csv](file:///Users/rajdeepsinghpanwar/Downloads/MandiFlow/market_coords.csv) using pandas and assert that "DHARASHIV" accurately landed on the `latitude/longitude` of Maharashtra.
