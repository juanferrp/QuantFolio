import pandas as pd
import numpy as np

def _validate_returns(returns: pd.DataFrame) -> None:
    """Validate a matrix of financial returns."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame.")

    if returns.shape[0] < 2:
        raise ValueError("At least two observations are required.")

    if returns.shape[1] == 0:
        raise ValueError("At least one asset is required.")

    if not returns.columns.is_unique:
        raise ValueError("Asset names must be unique.")

    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in returns.dtypes):
        raise TypeError("All returns must be numeric.")

    values = returns.to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(values).all():
        raise ValueError("Returns must be finite.")

def mean_returns(returns: pd.DataFrame) -> pd.Series:                 #Calcular la media muestral o rendimientos medios
    """Calculate the sample mean return of each asset."""
    _validate_returns(returns)
    return returns.mean(axis=0)

def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:           #calculamos la matriz de covarianzas
    """Calculate the sample covariance matrix."""
    _validate_returns(returns)
    R = returns.to_numpy(dtype=float)
    T = R.shape[0]
    mu = R.mean(axis=0)
    X = R - mu
    Sigma = (X.T @ X) / (T - 1)
    return pd.DataFrame(Sigma,index=returns.columns,columns=returns.columns)

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:                  #calculamos matriz de correlación
    """Calculate the sample correlation matrix."""
    Sigma = covariance_matrix(returns)
    sigma = np.sqrt(np.diag(Sigma.to_numpy()))
    if np.any(sigma == 0):
        raise ValueError("Correlation is undefined for zero-variance assets.")
    C = Sigma.to_numpy() / np.outer(sigma, sigma)
    return pd.DataFrame(C,index=Sigma.index,columns=Sigma.columns)



