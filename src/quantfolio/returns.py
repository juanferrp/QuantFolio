import numpy as np
import pandas as pd

def _validate_prices(prices: pd.Series) -> None:

    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be a pandas Series.")

    if len(prices) < 2:
        raise ValueError("At least two prices are required.")

    if not pd.api.types.is_numeric_dtype(prices.dtype):
        raise TypeError("Prices must be numeric.")

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Prices must be finite.")

    if (prices <= 0).any():
        raise ValueError("All prices must be strictly positive.")



def simple_returns(prices: pd.Series) -> pd.Series:     #Rendimiento simple R_t=P_t/P_{t-1}-1#
    _validate_prices(prices)
    return prices.pct_change().iloc[1:]
    

def log_returns(prices: pd.Series) -> pd.Series:        #Rendimiento logarítmico r_t= ln(P_t/P_{t-1}) #
    _validate_prices(prices)
    return np.log(prices / prices.shift(1)).iloc[1:]
