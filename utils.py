import torch
import torch.nn.functional as F

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

def estimate_sigma_mnist(model, x_train, y_train):
    """
    Estimate sigma^2 for MNIST classification using variance of cross-entropy loss.

    Args:
        model (nn.Module): Trained MNIST_CNN model.
        x_train (torch.Tensor): Training images (N, 1, 28, 28).
        y_train (torch.Tensor): Training labels (N,).

    Returns:
        torch.Tensor: Estimated standard deviation sigma.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x_train)  # Get raw class scores (logits)
        log_probs = F.log_softmax(logits, dim=1)  # Log-probabilities

        # Compute cross-entropy loss per sample
        loss_per_sample = F.nll_loss(log_probs, y_train, reduction="none")

        # Compute variance of loss values
        sigma_sq = torch.var(loss_per_sample)  # Variance of NLL loss
        sigma = torch.sqrt(sigma_sq)  # Standard deviation

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
