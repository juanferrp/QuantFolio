import numpy as np
import pandas as pd
from quantfolio.backtesting import walk_forward_backtest

def test_walk_forward_output():
    dates = pd.date_range("2025-01-01", periods=10)
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.01, 0.03,0.01, -0.02, 0.01, 0.02, 0.01],"B": [0.00, 0.01, 0.02, -0.01, 0.01,0.02, 0.01, -0.01, 0.00, 0.02]},index=dates)
    result, weights = walk_forward_backtest(returns,lookback=5,rebalance_every=2)

    assert result.index.equals(dates[5:])

    assert set(result.columns) == {"min_variance","equal_weight"}
   
    np.testing.assert_allclose( weights.sum(axis=1).to_numpy(),1.0)

    assert np.isfinite(result.to_numpy()).all()
