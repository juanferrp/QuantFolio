import numpy as np
import pandas as pd

def _validate_prices(prices: pd.Series | pd.DataFrame) -> None:

    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("Prices must be a pandas Series or DataFrame.")

    if len(prices) < 2:
        raise ValueError("At least two observations are required.")

    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 0:
            raise ValueError("At least one asset is required.")

        if not all(pd.api.types.is_numeric_dtype(dtype)
                   for dtype in prices.dtypes):
            raise TypeError("All prices must be numeric.")

    elif not pd.api.types.is_numeric_dtype(prices.dtype):
        raise TypeError("Prices must be numeric.")

    values = prices.to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(values).all():
        raise ValueError("All prices must be finite.")

    if (values <= 0).any():
        raise ValueError("All prices must be strictly positive.")

def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:     #Rendimiento simple R_t=P_t/P_{t-1}-1#
    _validate_prices(prices)
    return prices.pct_change(fill_method=None).iloc[1:]
    

def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:        #Rendimiento logarítmico r_t= ln(P_t/P_{t-1}) #
    _validate_prices(prices)
    return np.log(prices / prices.shift(1)).iloc[1:]
