import numpy as np
from scipy.stats import norm
from sklearn.metrics import root_mean_squared_error

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE) using sklearn."""
    return root_mean_squared_error(y_true, y_pred)


def compute_regression_nll(mu, sigma, y_true, eps=1e-6):
    """
    Compute Negative Log-Likelihood (NLL) for Bayesian regression.

    Parameters:
    - mu: np.ndarray of shape (N,) - Predicted means
    - sigma: np.ndarray of shape (N,) - Predicted standard deviations
    - y_true: np.ndarray of shape (N,) - True values
    - eps: float - Small value to prevent log(0)

    Returns:
    - nll: float - Mean Negative Log-Likelihood over all predictions
    """
    sigma = np.clip(sigma, a_min=eps, a_max=None)  # Avoid zero std
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + ((y_true - mu) ** 2) / (2 * sigma**2)
    return np.mean(nll)


def compute_sharpness(sigma):
    """
    Compute sharpness: average predictive standard deviation.

    Parameters:
    - sigma: np.ndarray of predicted std devs

    Returns:
    - sharpness: float
    """
    return np.mean(sigma)


def compute_coverage(mu, sigma, y_true, confidence=0.9):
    """
    Compute coverage: % of true values inside the predicted confidence interval.

    Parameters:
    - mu: np.ndarray of predicted means
    - sigma: np.ndarray of predicted std devs
    - y_true: np.ndarray of true targets
    - confidence: float (e.g., 0.9 for 90%)

    Returns:
    - coverage: float (fraction of y_true inside the interval)
    """
    z = norm.ppf(0.5 + confidence / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = ((y_true >= lower) & (y_true <= upper)).sum()
    return covered / len(y_true)

def compute_regression_ece(mu, sigma, y_true, alphas=None):
    """
    Compute Expected Calibration Error (ECE) for Bayesian regression models.

    Parameters:
    - mu: np.ndarray of shape (N,) - Predicted means
    - sigma: np.ndarray of shape (N,) - Predicted standard deviations
    - y_true: np.ndarray of shape (N,) - True values
    - alphas: List of confidence levels (e.g., [0.1, 0.2, ..., 0.9])

    Returns:
    - ece: float - Expected Calibration Error
    - coverage_list: list - Actual coverage per alpha
    """

    if alphas is None:
        alphas = np.linspace(0.1, 0.9, 9)

    N = len(mu)
    ece = 0.0
    coverage_list = []

    for alpha in alphas:
        z = norm.ppf(0.5 + alpha / 2)  # Symmetric interval around mean
        lower = mu - z * sigma
        upper = mu + z * sigma

        # Count how many true values fall within the interval
        covered = ((y_true >= lower) & (y_true <= upper)).sum()
        coverage = covered / N
        coverage_list.append(coverage)

        ece += abs(coverage - alpha)

    ece /= len(alphas)
    return ece, list(zip(alphas, coverage_list))

