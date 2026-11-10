import numpy as np   
import pandas as pd 
from quantfolio.statistics import covariance_matrix
from quantfolio.optimization import minimum_variance_portfolio

def walk_forward_backtest(returns: pd.DataFrame,lookback: int = 252,rebalance_every: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """*Backtesting* con ventana deslizante (*walk-forward*). 
    Nos devuelve
    -------
    portfolio_returns:
    Rentabilidades diarias de la cartera fuera de la muestra (*out-of-sample*). 

    weights_history:
    Ponderaciones objetivo en cada fecha de rebalanceo. 

    Supuestos
    -----------
    - Sin costes de transacción. 
    - Sin ventas en corto. 
    - Inversión total. 
    - Estrategia de «comprar y mantener» (*buy-and-hold*) entre fechas de rebalanceo."""
    
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a DataFrame.")

    if not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2.")

    if not isinstance(rebalance_every, int) or rebalance_every < 1:
        raise ValueError("rebalance_every must be positive.")

    if len(returns) <= lookback:
        raise ValueError("Insufficient observations.")

    if returns.shape[1] == 0 or not returns.columns.is_unique:
        raise ValueError("Asset columns must be nonempty and unique.")

    if (not returns.index.is_unique or not returns.index.is_monotonic_increasing):
        raise ValueError("Dates must be ordered and unique.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in returns.dtypes):
        raise TypeError("Returns must be numeric.")

    values = returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Returns must be finite.")

    if (values <= -1).any():
        raise ValueError("Daily returns must be greater than -1.")

    n = returns.shape[1]
    daily_results = []
    weights_history = []

    for start in range(lookback, len(returns), rebalance_every):

        # Información disponible antes del siguiente período de retención
        train = returns.iloc[start - lookback:start]
        sigma = covariance_matrix(train)
        optimal = minimum_variance_portfolio(sigma)
        target_weights = {"min_variance": optimal.reindex(returns.columns).to_numpy(),"equal_weight": np.ones(n) / n}

        # Observaciones futuras son solo usadas para evaluación 
        test = returns.iloc[start:start + rebalance_every]
        weights_history.append(pd.Series(target_weights["min_variance"],index=returns.columns,name=test.index[0]))
        block_results = {}

        for strategy, target in target_weights.items():
            current_weights = target.copy()
            realized_returns = []
            for _, row in test.iterrows():
                asset_returns = row.to_numpy(dtype=float)
                portfolio_return = current_weights @ asset_returns
                realized_returns.append(portfolio_return)

                # Las ponderaciones varían a medida que cambian los valores de los activos.
                current_weights = (current_weights * (1 + asset_returns)/ (1 + portfolio_return))
            block_results[strategy] = realized_returns
        daily_results.append(pd.DataFrame(block_results, index=test.index))

    portfolio_returns = pd.concat(daily_results)
    weights_history = pd.DataFrame(weights_history)
    return portfolio_returns, weights_history