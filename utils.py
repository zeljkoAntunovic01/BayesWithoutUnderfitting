import json
import numpy as np
import torch
from metrics import (
    compute_accuracy, compute_brier_score, compute_classification_ece_mce, compute_classification_nll, compute_confidence, compute_coverage, compute_multiclass_auroc, compute_predictive_entropy, compute_regression_ece, compute_regression_nll, compute_rmse, compute_sharpness
)

def compute_model_jacobian_params(model, x_train):
    """
    Computes the Jacobian of the model outputs w.r.t. model parameters manually using torch.autograd.grad().

    Args:
        model: PyTorch model (e.g., SineNet).
        x_train: Input data (torch.Tensor), shape (N, input_dim).

    Returns:
        J: Jacobian matrix (output_dim, num_params), shape (N, P).
    """
    model.eval()
    output = model(x_train)
    output = output.view(-1)  # Convert to shape (N,)

    num_params = sum(p.numel() for p in model.parameters())
    J = torch.zeros(output.shape[0], num_params)  # Shape: (N, P)

    for i in range(output.shape[0]):
        grad_output = torch.zeros_like(output)
        grad_output[i] = 1.0  # Select the i-th element to differentiate

        # Compute gradients with respect to model parameters
        grads = torch.autograd.grad(outputs=output, inputs=model.parameters(),
                                    grad_outputs=grad_output, retain_graph=True, allow_unused=True)

        # Flatten and concatenate gradients into the Jacobian
        grads = [g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, model.parameters())]
        J[i, :] = torch.cat(grads)

    return J

def compute_loss_jacobian_params(model, x_train, y_train, criterion):
    """
    Computes the Jacobian of the loss w.r.t. model parameters manually using torch.autograd.grad().

    Args:
        model: PyTorch model (e.g., SineNet).
        x_train: Input data (torch.Tensor), shape (N, input_dim).
        y_train: Target data (torch.Tensor), shape (N, output_dim).
        criterion: Loss function (e.g., nn.MSELoss()).	

    Returns:
        J: Jacobian matrix (output_dim, num_params), shape (N, P).
    """
    J_L_theta = []
    
    for x, y in zip(x_train, y_train):
        x = x.unsqueeze(0)  # Add batch dimension
        y = y.unsqueeze(0)
        
        model.zero_grad()  # Clear previous gradients
        y_pred = model(x) 
        loss = criterion(y_pred, y)
        
        # Compute gradients of the loss w.r.t. model parameters
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
        
        # Flatten and accumulate gradients
        J_L_theta.append(torch.cat([g.view(-1) for g in grads]))
    
    J_L_theta = torch.stack(J_L_theta)  # Shape: (N, P)
    return J_L_theta

def estimate_sigma_sine(model, x_train, y_train):
    """Estimate sigma^2 from dataset using residual variance."""
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train)
        residuals = y_train - y_pred
        sigma_sq = torch.mean(residuals**2)  # Estimate variance
        sigma = torch.sqrt(sigma_sq)  # Estimate standard deviation
    return sigma

def compute_model_jacobian_params_classifier(model, x_train):
    """
    Computes the Jacobian of the model outputs (logits) w.r.t. model parameters
    manually using torch.autograd.grad() for a classification model like MNIST CNN.

    Args:
        model: PyTorch classification model (e.g., MNIST_CNN).
        x_train: Input data (torch.Tensor), shape (N, 1, 28, 28).

    Returns:
        J: Jacobian matrix (N * num_classes, P), where:
           - N = number of samples
           - num_classes = number of output classes (10 for MNIST)
           - P = total number of model parameters
    """
    model.eval()
    
    # Move input to the same device as the model
    device = next(model.parameters()).device
    x_train = x_train.to(device)

    # Forward pass to get logits
    output = model(x_train)  # Shape: (N, num_classes)
    
    num_classes = output.shape[1]  # Should be 10 for MNIST
    num_samples = output.shape[0]  # N samples

    # Total number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())

    # Jacobian matrix storage
    J = torch.zeros(num_samples * num_classes, num_params, device=device)  # Shape: (N * 10, P)

    for i in range(num_samples):
        for j in range(num_classes):
            grad_output = torch.zeros_like(output)
            grad_output[i, j] = 1.0  # One-hot selection of the (i, j)th output

            # Compute gradient of logits[j] w.r.t. all model parameters
            grads = torch.autograd.grad(outputs=output, inputs=model.parameters(),
                                        grad_outputs=grad_output, retain_graph=True, allow_unused=True)

            # Flatten and concatenate gradients into the Jacobian
            grads = [g.view(-1) if g is not None else torch.zeros(p.numel(), device=device)
                     for g, p in zip(grads, model.parameters())]
            
            J[i * num_classes + j, :] = torch.cat(grads)  # Store row in Jacobian

    return J

def compute_loss_jacobian_params_classifier(model, x_train, y_train, criterion):
    """
    Computes the Jacobian of the per-datum loss w.r.t. model parameters
    for a classification model (e.g. FC_2D_Net on MNIST or 2D toy data).

    Args:
        model: PyTorch classification model.
        x_train: Input data (torch.Tensor), shape (N, D).
        y_train: Ground-truth labels (torch.Tensor), shape (N,).
        criterion: Loss function (e.g., nn.CrossEntropyLoss with reduction='sum').

    Returns:
        J_L: Jacobian matrix (N, P) where:
             - N = number of samples
             - P = number of model parameters
    """
    model.eval()
    device = next(model.parameters()).device
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    num_samples = x_train.shape[0]
    num_params = sum(p.numel() for p in model.parameters())
    J_L_theta = torch.zeros(num_samples, num_params, device=device)

    for i in range(num_samples):
        model.zero_grad()
        xi = x_train[i].unsqueeze(0)      # Shape: (1, D)
        yi = y_train[i].unsqueeze(0)      # Shape: (1,)
        
        output = model(xi)                # Shape: (1, num_classes)
        loss = criterion(output, yi)      # Scalar loss
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

        grads = [g.view(-1) for g in grads]
        J_L_theta[i] = torch.cat(grads)

    return J_L_theta



def sample_from_posterior(theta_map, covariance, num_samples=10, scale=0.1):
    """
    Samples parameters from the posterior using SVD for stability
    with added scaling to control uncertainty spread.
    Args:
        theta_map: Flattened MAP estimate of model parameters.
        covariance: Posterior covariance matrix.
        num_samples: Number of samples to draw.
        scale: Scaling factor for eigenvalues.
    Returns:
        samples: List of sampled parameter tensors.
    """
    U, S, V = torch.svd(covariance)
    print(f"Number of clamped eigenvalues: {(S < 1e-6).sum().item()}")
    S = torch.clamp(S, min=1e-6)  # Avoid near-zero eigenvalues
    L = U @ torch.diag(torch.sqrt(scale * S))  # Scale eigenvalues down

    samples = []

    for _ in range(num_samples):
        epsilon = torch.randn_like(theta_map)  # Sample from N(0, I)
        theta_sample = theta_map + L @ epsilon  # Apply scaled transformation
        samples.append(theta_sample)

    return samples

def disassemble_data_loader(data_loader):
    # Collect full data
    x_list, y_list = [], []
    for images, labels in data_loader:
        x_list.append(images)
        y_list.append(labels)

    # Convert full dataset to tensors
    x_train = torch.cat(x_list, dim=0)  # Full training images
    y_train = torch.cat(y_list, dim=0)  # Full training labels
    
    return x_train, y_train

def save_metrics_regression(y_test_pred, mean_pred, var_pred, y_test_true, path):
    # MAP metrics
    rmse_map = compute_rmse(y_test_true, y_test_pred)

    # Bayesian model metrics
    rmse_bayesian = compute_rmse(y_test_true, mean_pred)
    nll_bayesian = compute_regression_nll(mean_pred, np.sqrt(var_pred), y_test_true)
    ece_bayesian, coverage_per_alpha = compute_regression_ece(mean_pred, np.sqrt(var_pred), y_test_true)
    sharpness_bayesian = compute_sharpness(np.sqrt(var_pred))
    coverage__bayesian = compute_coverage(mean_pred, np.sqrt(var_pred), y_test_true)

    print(f"RMSE (MAP Model): {rmse_map:.4f}")
    print(f"RMSE (Bayesian Mean Model): {rmse_bayesian:.4f}")
    print(f"NLL (Bayesian Model): {nll_bayesian:.4f}")
    print(f"ECE (Bayesian Model): {ece_bayesian:.4f}")

    # Save the metrics
    metrics = {
        "MAP": {
            "RMSE": float(rmse_map)
        },
        "Bayesian": {
            "RMSE": float(rmse_bayesian),
            "NLL": float(nll_bayesian),
            "ECE": float(ece_bayesian),
            "Sharpness": float(sharpness_bayesian),
            "Coverage@90": float(coverage__bayesian),
            "Coverage_per_alpha": [
                {
                    "confidence": float(alpha),
                    "empirical_coverage": float(cov)
                }
                for alpha, cov in coverage_per_alpha
            ]
        }
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

def save_metrics_classification(y_test_pred_map, y_test_pred, mean_probs_pred, var_pred, y_test_true, path):
    # MAP metrics
    acc_map = compute_accuracy(y_test_true, y_test_pred_map)
    nll_map = compute_classification_nll(y_test_true, y_test_pred_map)
    ece_map, mce_map = compute_classification_ece_mce(y_test_true, y_test_pred_map, n_bins=10)
    brier_score_map = compute_brier_score(y_test_true, y_test_pred_map)
    confidence_map = compute_confidence(y_test_pred_map)
    predictive_entropy_map = compute_predictive_entropy(y_test_pred_map)
    auroc_per_class_map = compute_multiclass_auroc(y_test_true, y_test_pred_map)

    # Bayesian model metrics
    acc_bayesian = compute_accuracy(y_test_true, mean_probs_pred)
    nll_bayesian = compute_classification_nll(y_test_true, mean_probs_pred)
    ece_bayesian, mce_bayesian = compute_classification_ece_mce(y_test_true, mean_probs_pred, n_bins=10)
    brier_score_bayesian = compute_brier_score(y_test_true, mean_probs_pred)
    confidence_bayesian = compute_confidence(mean_probs_pred)
    predictive_entropy_bayesian = compute_predictive_entropy(mean_probs_pred)
    aurco_per_class_bayesian = compute_multiclass_auroc(y_test_true, mean_probs_pred)

    print(f"Accuracy (MAP Model): {acc_map:.4f}")
    print(f"Accuracy (Bayesian Mean Model): {acc_bayesian:.4f}")
    print(f"NLL (Bayesian Model): {nll_bayesian:.4f}")
    print(f"NLL (MAP Model): {nll_map:.4f}")
    print(f"ECE (Bayesian Model): {ece_bayesian:.4f}")
    print(f"ECE (MAP Model): {ece_map:.4f}")

    # Save the metrics
    metrics = {
        "MAP": {
            "Accuracy": float(acc_map),
            "NLL": float(nll_map),
            "ECE": float(ece_map),
            "MCE": float(mce_map),
            "Brier Score": float(brier_score_map),
            "Confidence": float(confidence_map),
            "Predictive Entropy": float(predictive_entropy_map),
            "AUROC per class": [
                {
                    "class": str(key),
                    "auroc": float(value)
                }
                for key, value in auroc_per_class_map.items()
            ]
        },
        "Bayesian": {
            "Accuracy": float(acc_bayesian),
            "NLL": float(nll_bayesian),
            "ECE": float(ece_bayesian),
            "MCE": float(mce_bayesian),
            "Brier Score": float(brier_score_bayesian),
            "Confidence": float(confidence_bayesian),
            "Predictive Entropy": float(predictive_entropy_bayesian),
            "AUROC per class": [
                {
                    "class": str(key),
                    "auroc": float(value)
                }
                for key, value in aurco_per_class_bayesian.items()
            ]
        }
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics