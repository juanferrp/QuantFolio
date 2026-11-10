import pandas as pd 
import numpy as np   
from scipy.optimize import minimize

def minimum_variance_portfolio(covariance: pd.DataFrame) -> pd.Series:
    """Encuentra el portafolio con mínima varianza.
    Restricciones:
        suma de los pesos = 1
        los pesos son >= 0"""
        
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("Covariance must be a pandas DataFrame.")

    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be nonempty and square.")

    if not covariance.index.is_unique or not covariance.columns.is_unique:
        raise ValueError("Asset names must be unique.")

    if not covariance.index.equals(covariance.columns):
        raise ValueError("Row and column asset labels must match.")

    sigma = covariance.to_numpy(dtype=float)

    if not np.isfinite(sigma).all():
        raise ValueError("Covariance matrix must be finite.")

    if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=0):
        raise ValueError("Covariance matrix must be symmetric.")

    if np.linalg.eigvalsh(sigma).min() < -1e-10:
        raise ValueError("Covariance matrix must be positive semidefinite.")

    n = sigma.shape[0]

    def objective(w):                                                                                 # Función objetivo: varianza de portafolio 
        return w @ sigma @ w

    def gradient(w):                                                                                  #Gradiente del objetivo
        return 2 * sigma @ w

    w0 = np.ones(n) / n                                                                               #portafolio Inicial, con pesos iguales
    bounds = [(0.0, 1.0)] * n                                                                         #Prohibición de ventas en corto
    constraints = [{"type": "eq","fun": lambda w: np.sum(w) - 1,"jac": lambda w: np.ones_like(w)}]    #Inversión total y restricciones

    result = minimize(fun=objective,x0=w0,jac=gradient,method="SLSQP",bounds=bounds,constraints=constraints,options={"ftol": 1e-12, "maxiter": 2000})

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x,index=covariance.columns,name="weights")

def mean_variance_portfolio(expected_returns: pd.Series,covariance: pd.DataFrame,risk_aversion: float) -> pd.Series:
    """Maximizar el rendimiento esperado menos una penalización por riesgo. 
        Los rendimientos esperados y la covarianza deben utilizar
        el mismo horizonte temporal (anual, en nuestros ejemplos)."""

    if not isinstance(expected_returns, pd.Series):
        raise TypeError("Expected returns must be a pandas Series.")

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("Covariance must be a pandas DataFrame.")

    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Risk aversion must be positive and finite.")

    assets = expected_returns.index

    if not assets.is_unique:
        raise ValueError("Asset names must be unique.")

    if (not covariance.index.is_unique or not covariance.columns.is_unique or set(assets) != set(covariance.index) or set(assets) != set(covariance.columns)):
        raise ValueError("Assets must match the covariance matrix.")

    # Alineamos los assets por nombre y no por posición
    covariance = covariance.loc[assets, assets]
    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)

    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("Inputs must be finite.")

    if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=0):
        raise ValueError("Covariance must be symmetric.")

    if np.linalg.eigvalsh(sigma).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite.")

    n = len(mu)

    def objective(w):
        return -(mu @ w - risk_aversion / 2 * (w @ sigma @ w))

    def gradient(w):
        return -mu + risk_aversion * sigma @ w

    w0 = np.ones(n) / n

    constraints = [{"type": "eq","fun": lambda w: np.sum(w) - 1,"jac": lambda w: np.ones_like(w)}]
    result = minimize(objective,w0,method="SLSQP",jac=gradient,bounds=[(0, 1)] * n,constraints=constraints,options={"ftol": 1e-12, "maxiter": 2000})

    if not result.success:
        raise RuntimeError(result.message)

    return pd.Series(result.x, index=assets, name="weights")