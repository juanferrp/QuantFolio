import numpy as np
import pandas as pd

from quantfolio.statistics import (mean_returns,covariance_matrix,correlation_matrix)

def sample_returns():
    return pd.DataFrame({"SPY": [0.10, 0.06, 0.02],"BND": [0.02, 0.00, -0.02],"GLD": [-0.01, 0.03, 0.02]})

def test_mean_returns():
    R = sample_returns()
    result = mean_returns(R)
    np.testing.assert_allclose(result.to_numpy(),[0.06, 0.00, 0.013333333333333334])

def test_covariance_matches_pandas():
    R = sample_returns()
    result = covariance_matrix(R)
    expected = R.cov()
    np.testing.assert_allclose(result.to_numpy(),expected.to_numpy())

def test_covariance_is_symmetric():
    Sigma = covariance_matrix(sample_returns()).to_numpy()
    np.testing.assert_allclose(Sigma, Sigma.T)

def test_covariance_is_positive_semidefinite():
    Sigma = covariance_matrix(sample_returns()).to_numpy()
    eigenvalues = np.linalg.eigvalsh(Sigma)
    assert np.all(eigenvalues >= -1e-12)


def test_correlation_matches_pandas():
    R = sample_returns()
    result = correlation_matrix(R)
    expected = R.corr()
    np.testing.assert_allclose(result.to_numpy(),expected.to_numpy())