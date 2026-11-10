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