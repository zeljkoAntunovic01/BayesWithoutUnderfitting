import torch
import numpy as np
import matplotlib.pyplot as plt

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

def estimate_sigma(model, x_train, y_train):
    """Estimate sigma^2 from dataset using residual variance."""
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train)
        residuals = y_train - y_pred
        sigma_sq = torch.mean(residuals**2)  # Estimate variance
        sigma = torch.sqrt(sigma_sq)  # Estimate standard deviation
    return sigma
