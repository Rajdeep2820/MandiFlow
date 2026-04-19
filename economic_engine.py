import numpy as np

# Shock Types Map (Aligned with simulator.py)
SHOCK_NONE = 0
SHOCK_CLIMATIC = 1
SHOCK_LOGISTICS = 2
SHOCK_POLICY_UP = 3    # Import Ban / Export Allowed (Prices Rise)
SHOCK_POLICY_DOWN = 4  # Export Ban / Import Allowed (Prices Fall)

def apply_economic_constraints(
    forecast_prices: list[float], 
    forecast_dirs: list[float], 
    current_price: float, 
    shock_type: int, 
    is_epicenter: bool
) -> tuple[list[float], list[float]]:
    """
    Enforces rigid macroeconomic laws on top of the GCN-LSTM outputs.
    Guarantees the model cannot hallucinate contradictory price movements.
    """
    prices = np.array(forecast_prices, dtype=float)
    dirs = np.array(forecast_dirs, dtype=float)

    if shock_type == SHOCK_POLICY_DOWN:
        # Export Ban / Import Allow -> Supply increases -> Prices must drop or stay flat
        prices = np.clip(prices, a_min=0, a_max=current_price)
        dirs = np.where(prices <= current_price, np.minimum(dirs, 0.49), dirs)

    elif shock_type == SHOCK_POLICY_UP:
        # Export Allow / Import Ban -> Supply decreases -> Prices must rise
        prices = np.clip(prices, a_min=current_price, a_max=None)
        dirs = np.where(prices >= current_price, np.maximum(dirs, 0.51), dirs)

    elif shock_type == SHOCK_CLIMATIC:
        # Crop Destruction / Flood / Drought -> Supply decreases -> Prices must rise
        prices = np.clip(prices, a_min=current_price, a_max=None)
        dirs = np.where(prices >= current_price, np.maximum(dirs, 0.51), dirs)

    elif shock_type == SHOCK_LOGISTICS:
        if is_epicenter:
            # Trapped localized supply (e.g. Strike at origin) -> Prices drop or flatline
            prices = np.clip(prices, a_min=0, a_max=current_price)
            dirs = np.where(prices <= current_price, np.minimum(dirs, 0.49), dirs)
        else:
            # Destination shortage -> Goods not arriving -> Prices rise
            prices = np.clip(prices, a_min=current_price, a_max=None)
            dirs = np.where(prices >= current_price, np.maximum(dirs, 0.51), dirs)

    return prices.tolist(), dirs.tolist()
