import pandas as pd   
import numpy as np
from quantfolio.optimization import minimum_variance_portfolio
from quantfolio.optimization import mean_variance_portfolio

def test_analytical_solution():
    sigma=pd.DataFrame([[0.04,0],[0,0.01]], index=["A","B"], columns=["A","B"])
    weights= minimum_variance_portfolio(sigma)
    np.testing.assert_allclose(weights.to_numpy(), [0.2,0.8], atol=1e-5)

def test_portfolio_constraints():
    sigma=pd.DataFrame([[0.04, 0.01, 0.005],[0.01, 0.02, 0.003],[0.005, 0.003, 0.01]], index=["A","B","C"], columns=["A","B","C"])
    weights=minimum_variance_portfolio(sigma)
    np.testing.assert_allclose(weights.sum(),1.0,atol=1e-8)
    assert np.all(weights >= -1e-8)
    assert np.all(weights <= 1 + 1e-8)

def test_optimized_risk_not_greater_than_equal_weights():
    sigma = pd.DataFrame([[0.04, 0],[0, 0.01]],index=["A", "B"],columns=["A", "B"])
    optimal = minimum_variance_portfolio(sigma).to_numpy()
    equal = np.array([0.5, 0.5])
    matrix = sigma.to_numpy()
    optimal_risk = optimal @ matrix @ optimal
    equal_risk = equal @ matrix @ equal
    assert optimal_risk <= equal_risk + 1e-10
    
def test_mean_variance_analytical_solution():
    mu = pd.Series([0.12, 0.06], index=["A", "B"])
    sigma = pd.DataFrame([[0.04, 0],[0, 0.01]],index=["A", "B"],columns=["A", "B"])
    weights = mean_variance_portfolio(mu, sigma, 4)
    np.testing.assert_allclose(weights.to_numpy(),[0.5, 0.5],atol=1e-5)

def test_mean_variance_constraints():
    mu = pd.Series([0.10, 0.05], index=["A", "B"])
    sigma = pd.DataFrame([[0.04, 0.01],[0.01, 0.02]],index=["A", "B"],columns=["A", "B"])
    weights = mean_variance_portfolio(mu, sigma, 3)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-8)
    assert np.all(weights >= -1e-8)
    assert np.all(weights <= 1 + 1e-8)
