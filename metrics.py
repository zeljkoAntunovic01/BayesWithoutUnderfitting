import numpy as np
from scipy.stats import norm
from sklearn.metrics import root_mean_squared_error, roc_auc_score

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE) using sklearn."""
    return root_mean_squared_error(y_true, y_pred)

def compute_accuracy(y_true, y_pred_probs):
    """Compute classification accuracy."""
    y_pred = np.argmax(y_pred_probs, axis=1)
    return np.mean(y_true == y_pred)

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

def compute_classification_nll(y_true, y_pred_probs, eps=1e-8):
    """NLL for classification with predicted probabilities"""
    log_probs = np.log(np.clip(y_pred_probs[np.arange(len(y_true)), y_true], eps, 1.0))
    return -np.mean(log_probs)


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

def compute_classification_ece_mce(y_true, y_pred_probs, n_bins=15):
    """
    Compute both ECE and MCE (Expected and Maximum Calibration Error).
    
    Returns:
    - ece: float
    - mce: float
    """
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            gap = abs(bin_conf - bin_acc)
            ece += gap * np.sum(mask) / len(y_true)
            mce = max(mce, gap)
    
    return ece, mce


def compute_brier_score(y_true, y_pred_probs):
    """
    Compute Brier Score for multiclass classification.

    Parameters:
    - y_true: (N,) int array of true class indices
    - y_pred_probs: (N, C) array of predicted class probabilities

    Returns:
    - brier: float, mean Brier score
    """
    N, C = y_pred_probs.shape
    y_true_onehot = np.zeros_like(y_pred_probs)
    y_true_onehot[np.arange(N), y_true] = 1
    return np.mean(np.sum((y_pred_probs - y_true_onehot) ** 2, axis=1))

def compute_predictive_entropy(y_pred_probs, eps=1e-8):
    """
    Compute predictive entropy for each sample.
    
    Parameters:
    - y_pred_probs: (N, C) array of predicted class probabilities

    Returns:
    - entropies: (N,) array of entropy values per sample
    """
    clipped_probs = np.clip(y_pred_probs, eps, 1.0)
    entropies = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)
    return np.mean(entropies)

def compute_confidence(y_pred_probs):
    """
    Compute average confidence of predictions.
    
    Parameters:
    - y_pred_probs: (N, C) array of predicted class probabilities
    
    Returns:
    - confidence: float, average max predicted probability across samples
    """
    confidences = np.max(y_pred_probs, axis=1)  # confidence per sample
    return np.mean(confidences)

def compute_multiclass_auroc(y_true, y_pred_probs):
    """
    Compute per-class AUROC using one-vs-rest strategy.

    Parameters:
    - y_true: (N,) array of int true labels
    - y_pred_probs: (N, C) array of predicted class probabilities

    Returns:
    - auroc_per_class: dict[class_index] = AUROC
    """
    N, C = y_pred_probs.shape
    y_true_onehot = np.zeros_like(y_pred_probs)
    y_true_onehot[np.arange(N), y_true] = 1

    auroc_per_class = {}
    for c in range(C):
        try:
            score = roc_auc_score(y_true_onehot[:, c], y_pred_probs[:, c])
        except ValueError:
            score = float('nan')  # Handles class imbalance edge cases
        auroc_per_class[c] = score

    return auroc_per_class
