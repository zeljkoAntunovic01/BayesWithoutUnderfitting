import torch.nn as nn
import torch
import numpy as np

from utils import compute_model_jacobian_params, compute_loss_jacobian_params, estimate_sigma_sine

class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

#TODO: Increasing alpha from 1 to 10 made the model not underfit. Why?
def compute_q_lla(model, x_train, y_train, alpha=10.0):
    """
    Computes the q_LLA posterior for a trained model.
    
    Args:
        model: Trained SineNet model.
        x_train: Training inputs (torch.Tensor).
        y_train: Training outputs (torch.Tensor).
        alpha: Prior precision term (controls posterior variance).
    
    Returns:
        theta_map: Flattened MAP estimate of model parameters.
        covariance: (alpha I + GGN)^(-1), the approximate posterior covariance.
    """
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    
    J = compute_model_jacobian_params(model, x_train)
    print("Max Jacobian:", torch.max(J).item())
    print("Min Jacobian:", torch.min(J).item())

    sigma = estimate_sigma_sine(model, x_train, y_train)  # Estimate sigma
    GGN = (1 / sigma**2) * (J.T @ J)
    identity = torch.eye(GGN.shape[0])
    covariance = torch.inverse(alpha * identity + GGN)

    print("Covariance Diagonal Min:", torch.min(covariance.diagonal()).item())
    print("Covariance Diagonal Max:", torch.max(covariance.diagonal()).item())

    return theta_map, covariance

#TODO: When I left Alpha at 10, the model was super underfit. 50 seemed to work better, but why?
def compute_q_proj(model, x_train, y_train, alpha=50.0):
    """
    Computes the q_proj posterior for a trained model using the null space of the GGN matrix.
    
    Args:
        model: Trained SineNet model.
        x_train: Training inputs (torch.Tensor).
        y_train: Training outputs (torch.Tensor).
        alpha: Prior precision term (controls posterior variance).
    
    Returns:
        theta_map: Flattened MAP estimate of model parameters.
        projected_covariance: The approximate posterior covariance using the projected method.
    """
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    
    J = compute_model_jacobian_params(model, x_train)
    print("Max Jacobian:", torch.max(J).item())
    print("Min Jacobian:", torch.min(J).item())

    sigma = estimate_sigma_sine(model, x_train, y_train)  # Estimate sigma
    GGN = (1 / sigma**2) * (J.T @ J)
    eigenvalues, V = torch.linalg.eigh(GGN)
    
    #TODO: Why do we have to clamp? Why are the values so high?
    eigenvalues = torch.clamp(eigenvalues, max=100) 
    print("Eigenvalues Min:", torch.min(eigenvalues).item())
    print("Eigenvalues Max:", torch.max(eigenvalues).item())

    null_mask = (eigenvalues <= 1e-2).float() 
    print(f"Number of null space dimensions: {null_mask.sum().item()} / {eigenvalues.numel()}")
    I_p = torch.eye(GGN.shape[0])
    projection_matrix = I_p - (V @ (1 - null_mask)) @ V.T
    projected_covariance = (1.0 / alpha) * projection_matrix

    print("Projected Covariance Diagonal Min:", torch.min(projected_covariance.diagonal()).item())
    print("Projected Covariance Diagonal Max:", torch.max(projected_covariance.diagonal()).item())

    return theta_map, projected_covariance

def compute_q_loss(model, x_train, y_train, alpha=10.0):
    """
    Computes the q_loss posterior for a trained model using the Loss-Jacobian approach.
    
    Args:
        model: Trained SineNet model.
        x_train: Training inputs (torch.Tensor).
        y_train: Training outputs (torch.Tensor).
        alpha: Prior precision term (controls posterior variance).
    
    Returns:
        theta_map: Flattened MAP estimate of model parameters.
        projected_covariance: The approximate posterior covariance using the loss-projected method.
    """
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    criterion = nn.MSELoss(reduction='sum')
    J_L_theta = compute_loss_jacobian_params(model, x_train, y_train, criterion)
    print("Max Loss-Jacobian:", torch.max(J_L_theta).item())
    print("Min Loss-Jacobian:", torch.min(J_L_theta).item())

    sigma = estimate_sigma_sine(model, x_train, y_train)  # Estimate sigma
    GGN = (1 / sigma**2) * (J_L_theta.T @ J_L_theta)
    eigenvalues, V = torch.linalg.eigh(GGN)
    null_mask = (eigenvalues <= 1e-6).float()
    print(f"Number of null space dimensions: {null_mask.sum().item()} / {eigenvalues.numel()}")

    I_p = torch.eye(GGN.shape[0])
    projection_matrix = I_p - (V @ torch.diag(1 - null_mask) @ V.T)

    projected_covariance = (1.0 / alpha) * projection_matrix

    print("Projected Covariance Diagonal Min:", torch.min(projected_covariance.diagonal()).item())
    print("Projected Covariance Diagonal Max:", torch.max(projected_covariance.diagonal()).item())

    return theta_map, projected_covariance

def bayesian_prediction(model, theta_samples, x_test):
    """
    Makes Bayesian predictions by averaging over posterior samples.

    Args:
        model: Neural network model (SineNet).
        theta_samples: List of sampled parameter tensors.
        x_test: Test inputs (torch.Tensor).

    Returns:
        mean_pred: Mean of sampled predictions.
        var_pred: Variance of sampled predictions.
    """
    model.eval()
    predictions = []

    for theta_sample in theta_samples:
        torch.nn.utils.vector_to_parameters(theta_sample, model.parameters())  # Load new parameters
        y_pred = model(x_test).detach().numpy()
        predictions.append(y_pred)

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    var_pred = np.var(predictions, axis=0) 

    return mean_pred, var_pred