import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils import compute_model_jacobian_params, compute_loss_jacobian_params, compute_model_jacobian_params_classifier, estimate_sigma_mnist

class MNIST_FC(nn.Module):
    def __init__(self, hidden_size=28):
        super(MNIST_FC, self).__init__()

        # Fully Connected Layers
        self.fc1 = nn.Linear(28 * 28, hidden_size)  # Input: 784 -> hidden
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden -> Hidden
        self.fc3 = nn.Linear(hidden_size, 10)  # Hidden -> Output (10 classes)

        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image (28x28 -> 784)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No Softmax (handled by CrossEntropyLoss)
        return x

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Output: 28x28x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: 28x28x64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples: 14x14

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Flattened features from conv layers
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x 

def compute_q_lla(model, x_train, y_train, alpha=10.0):
    """
    Computes the q_LLA posterior for a trained CNN model.

    Args:
        model: Trained MNIST_CNN model.
        x_train: Training inputs (torch.Tensor).
        y_train: Training outputs (torch.Tensor).
        alpha: Prior precision term (controls posterior variance).

    Returns:
        theta_map: Flattened MAP estimate of model parameters.
        covariance: (alpha I + GGN)^(-1), the approximate posterior covariance.
    """
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    
    J = compute_model_jacobian_params_classifier(model, x_train)
    print("Max Jacobian:", torch.max(J).item())
    print("Min Jacobian:", torch.min(J).item())

    sigma = estimate_sigma_mnist(model, x_train, y_train)  # Estimate sigma
    GGN = (1 / sigma**2) * (J.T @ J)
    identity = torch.eye(GGN.shape[0], device=GGN.device)
    covariance = torch.inverse(alpha * identity + GGN)

    print("Covariance Diagonal Min:", torch.min(covariance.diagonal()).item())
    print("Covariance Diagonal Max:", torch.max(covariance.diagonal()).item())

    return theta_map, covariance

def bayesian_prediction(model, theta_samples, x_test):
    """
    Makes Bayesian predictions by averaging over posterior samples for a classification model (MNIST).

    Args:
        model (torch.nn.Module): Neural network model (e.g., MNIST_CNN).
        theta_samples (list of torch.Tensor): List of sampled parameter tensors.
        x_test (torch.Tensor): Test inputs, shape (N, 1, 28, 28).

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
