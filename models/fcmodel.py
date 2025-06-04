import torch.nn as nn
import torch.nn.functional as F 
import torch
import numpy as np

from utils import compute_loss_jacobian_params_classifier, compute_model_jacobian_params_classifier, get_param_vector_tools
from torch.func import functional_call

class FC_2D_Net(nn.Module):
    def __init__(self, hidden_units=64, n_classes=4):
        super(FC_2D_Net, self).__init__()

        # Fully Connected Layers
        self.fc1 = nn.Linear(2, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Raw logits
        return x
    
def compute_q_lla(model, x_train, y_train, alpha=2.0):
    """
    Computes the q_LLA posterior for a trained classifier model.

    Args:
        model: Trained FC_2D_Net model.
        x_train: Training inputs (torch.Tensor).
        y_train: Training outputs (torch.Tensor).
        alpha: Prior precision term (controls posterior variance).

    Returns:
        theta_map: Flattened MAP estimate of model parameters.
        covariance: (alpha I + GGN)^(-1), the approximate posterior covariance.
    """
    device = next(model.parameters()).device
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    N = x_train.shape[0]

    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    
    logits = model(x_train)                      # (N, O)
    probs = torch.softmax(logits, dim=1)         # (N, O)
    O = probs.shape[1]

    J = compute_model_jacobian_params_classifier(model, x_train)
    print("Max Jacobian:", torch.max(J).item())
    print("Min Jacobian:", torch.min(J).item())

    # ---- Construct block-diagonal Hessian matrix H ∈ (N·O, N·O) ----
    H_blocks = []
    for i in range(N):
        p = probs[i].unsqueeze(1)                # (O, 1)
        Hi = torch.diag_embed(probs[i]) - p @ p.T  # (O, O)
        H_blocks.append(Hi)

    H = torch.block_diag(*H_blocks)              # (N·O, N·O)

    GGN = (J.T @ H @ J)
    identity = torch.eye(GGN.shape[0], device=GGN.device)
    covariance = torch.inverse(alpha * identity + GGN)

    print("Covariance Diagonal Min:", torch.min(covariance.diagonal()).item())
    print("Covariance Diagonal Max:", torch.max(covariance.diagonal()).item())

    return theta_map, covariance

def compute_q_proj(model, x_train, y_train, alpha=10.0):
    """
    Computes the q_proj posterior for a trained classifier model using the null space of the GGN matrix.
    
    Args:
        model: Trained classifier model.
        x_train: Training inputs (torch.Tensor).
        y_train: Training outputs (torch.Tensor).
        alpha: Prior precision term (controls posterior variance).
    
    Returns:
        theta_map: Flattened MAP estimate of model parameters.
        projected_covariance: The approximate posterior covariance using the projected method.
    """
    device = next(model.parameters()).device
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    N = x_train.shape[0]

    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    
    logits = model(x_train)                      # (N, O)
    probs = torch.softmax(logits, dim=1)         # (N, O)
    O = probs.shape[1]

    J = compute_model_jacobian_params_classifier(model, x_train)
    print("Max Jacobian:", torch.max(J).item())
    print("Min Jacobian:", torch.min(J).item())

    # ---- Construct block-diagonal Hessian matrix H ∈ (N·O, N·O) ----
    H_blocks = []
    for i in range(N):
        p = probs[i].unsqueeze(1)                # (O, 1)
        Hi = torch.diag_embed(probs[i]) - p @ p.T  # (O, O)
        H_blocks.append(Hi)

    H = torch.block_diag(*H_blocks)              # (N·O, N·O)

    GGN = (J.T @ H @ J)
    identity = torch.eye(GGN.shape[0], device=GGN.device)
    eigenvalues, V = torch.linalg.eigh(identity + GGN)
    
    #TODO: Why do we have to clamp? Why are the values so high?
    eigenvalues = torch.clamp(eigenvalues, max=100) 
    print("Eigenvalues Min:", torch.min(eigenvalues).item())
    print("Eigenvalues Max:", torch.max(eigenvalues).item())

    null_mask = (eigenvalues <= 1e-2 + 1).float() 
    print(f"Number of null space dimensions: {null_mask.sum().item()} / {eigenvalues.numel()}")
    I_p = torch.eye(GGN.shape[0])
    projection_matrix = I_p - (V @ (1 - null_mask)) @ V.T
    projected_covariance = (1.0 / alpha) * projection_matrix

    print("Projected Covariance Diagonal Min:", torch.min(projected_covariance.diagonal()).item())
    print("Projected Covariance Diagonal Max:", torch.max(projected_covariance.diagonal()).item())

    return theta_map, projected_covariance

def compute_q_loss(model, x_train, y_train, alpha=10.0):
    """
    Computes the q_loss posterior for a trained classifier model using the null space of the stacked loss gradients.

    Args:
        model: Trained classifier model.
        x_train: Training inputs (torch.Tensor).
        y_train: Training outputs (torch.Tensor).
        alpha: Prior precision term (controls posterior variance).
        loss_fn: Loss function used for training (default: cross-entropy).

    Returns:
        theta_map: Flattened MAP estimate of model parameters.
        loss_projected_covariance: Approximate posterior covariance using the loss-projection method.
    """
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    criterion = nn.CrossEntropyLoss(reduction='sum')

    J_L_theta = compute_loss_jacobian_params_classifier(model, x_train, y_train, criterion)
    print("Max Jacobian:", torch.max(J_L_theta).item())
    print("Min Jacobian:", torch.min(J_L_theta).item())

    A = (J_L_theta.T @ J_L_theta)
    identity = torch.eye(A.shape[0], device=A.device)
    eigenvalues, V = torch.linalg.eigh(identity + A)
    
    eigenvalues = torch.clamp(eigenvalues, max=100) 
    print("Eigenvalues Min:", torch.min(eigenvalues).item())
    print("Eigenvalues Max:", torch.max(eigenvalues).item())

    null_mask = (eigenvalues <= (1e-2 + 1)).float() 
    print(f"Number of null space dimensions: {null_mask.sum().item()} / {eigenvalues.numel()}")
    I_p = torch.eye(A.shape[0])
    projection_matrix = I_p - (V @ (1 - null_mask) @ V.T)
    projected_covariance = (1.0 / alpha) * projection_matrix

    print("Projected Covariance Diagonal Min:", torch.min(projected_covariance.diagonal()).item())
    print("Projected Covariance Diagonal Max:", torch.max(projected_covariance.diagonal()).item())

    return theta_map, projected_covariance

def bayesian_prediction(model, theta_samples, x_test):
    """
    Makes Bayesian predictions by averaging over posterior samples for a classification model (MNIST).

    Args:
        model (torch.nn.Module): Neural network model (e.g., FC_2D_Net).
        theta_samples (list of torch.Tensor): List of sampled parameter tensors.
        x_test (torch.Tensor): Test inputs, shape (N, 2).

    Returns:
        mean_pred (np.ndarray): Mean of sampled class probabilities, shape (N, num_classes).
        var_pred (np.ndarray): Variance of sampled class probabilities, shape (N, num_classes).
    """
    model.eval()
    device = next(model.parameters()).device  # Ensure data is on the correct device
    x_test = x_test.to(device)

    predictions = []

    for theta_sample in theta_samples:
        # Load new parameters into model
        torch.nn.utils.vector_to_parameters(theta_sample, model.parameters())

        # Compute predictions (logits) and convert to probabilities using softmax
        with torch.no_grad():
            logits = model(x_test)  # Shape: (N, num_classes)
            probs = F.softmax(logits, dim=1)  # Convert logits to class probabilities

        predictions.append(probs.cpu().numpy())  # Store in CPU memory to avoid GPU overflow

    predictions = np.array(predictions)  # Shape: (num_samples, N, num_classes)

    # Compute mean and variance over the sampled weights
    mean_pred = np.mean(predictions, axis=0)  # Shape: (N, num_classes)
    var_pred = np.var(predictions, axis=0)  # Shape: (N, num_classes)

    return mean_pred, var_pred

def bayesian_prediction_alt(model, theta_samples, x_test):
    """
    Makes Bayesian predictions by averaging over posterior samples for a classification model (MNIST).

    Args:
        model (torch.nn.Module): Neural network model (e.g., FC_2D_Net).
        theta_samples (list of torch.Tensor): List of sampled parameter tensors.
        x_test (torch.Tensor): Test inputs, shape (N, 2).

    Returns:
        mean_pred (np.ndarray): Mean of sampled class probabilities, shape (N, num_classes).
        var_pred (np.ndarray): Variance of sampled class probabilities, shape (N, num_classes).
    """
    model.eval()
    device = next(model.parameters()).device  # Ensure data is on the correct device
    params = dict(model.named_parameters())
    _, unflatten_params = get_param_vector_tools(params)

    # Define stateless model_fn
    def model_fn(theta_unflattened, x):
        return functional_call(model, theta_unflattened, x)
    
    predictions = []
    x_test = x_test.to(device)

    for theta_sample in theta_samples:
        # Load new parameters into model
        theta_flattened = unflatten_params(theta_sample)  # Unflatten the sampled parameters

        # Compute predictions (logits) and convert to probabilities using softmax
        with torch.no_grad():
            logits = model_fn(theta_flattened, x_test)  # Shape: (N, num_classes)
            probs = F.softmax(logits, dim=1)  # Convert logits to class probabilities

        predictions.append(probs.cpu().numpy())  # Store in CPU memory to avoid GPU overflow

    predictions = np.array(predictions)  # Shape: (num_samples, N, num_classes)

    # Compute mean and variance over the sampled weights
    mean_pred = np.mean(predictions, axis=0)  # Shape: (N, num_classes)
    var_pred = np.var(predictions, axis=0)  # Shape: (N, num_classes)

    return mean_pred, var_pred
