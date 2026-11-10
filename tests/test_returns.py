import numpy as np
import pandas as pd
import pytest
from quantfolio.returns import simple_returns, log_returns

@pytest.mark.parametrize("values", [[100],[100, 0, 120],[100, -10, 120],[100, np.nan, 120],[100, np.inf, 120],])

def test_invalid_prices(values):
    prices = pd.Series(values, dtype=float)

    with pytest.raises(ValueError):
        simple_returns(prices)

    with pytest.raises(ValueError):
        log_returns(prices)

def test_simple_returns():
    prices=pd.Series([100,110,99])
    result = simple_returns(prices)
    expected=np.array([0.1,-0.1])
    np.testing.assert_allclose(result.to_numpy(), expected)
    
def test_log_returns():
    prices=pd.Series([100,110,99])
    result = log_returns(prices)
    expected = np.array([np.log(1.1),np.log(0.9)])
    np.testing.assert_allclose(result.to_numpy(), expected)
    
def test_returns_preserve_dates():
    dates = pd.date_range("2025-01-01", periods=3)
    prices = pd.Series([100, 110, 99],index=dates)
    result = simple_returns(prices)
    expected_dates = dates[1:]
    pd.testing.assert_index_equal(result.index,expected_dates)   

def test_scale_invariance():                                                           #Test invarianza de la escala
    prices = pd.Series([100, 110, 99, 125])
    scaled_prices = prices * 1000
    np.testing.assert_allclose(simple_returns(prices),simple_returns(scaled_prices))
    np.testing.assert_allclose(log_returns(prices),log_returns(scaled_prices))
    
def test_simple_log_relationship():                                                    #Se debe cumplir R_t=e^{r_t}-1
    prices = pd.Series([100, 120, 90, 105])
    simple = simple_returns(prices)
    logarithmic = log_returns(prices)
    reconstructed = np.expm1(logarithmic)
    np.testing.assert_allclose(simple, reconstructed)
    
def test_cumulative_returns():                                                          #Rendimiento acomulado
    prices = pd.Series([100, 110, 99, 120])                                             #Para R_1,...,R_n 
    returns = simple_returns(prices)                                                    #\prod_{t=1}^n(1+R_t)=P_n/P_0
    cumulative_growth = np.prod(1 + returns)
    expected_growth = prices.iloc[-1] / prices.iloc[0]
    np.testing.assert_allclose( cumulative_growth,expected_growth)