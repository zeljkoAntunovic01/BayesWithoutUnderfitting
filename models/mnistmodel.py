import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
def bayesian_prediction(model, theta_samples, test_loader):
    """
    Makes Bayesian predictions by averaging over posterior samples for a classification model (MNIST).

    Args:
        model (torch.nn.Module): Neural network model (e.g., MNIST_Net).
        theta_samples (list of torch.Tensor): List of sampled parameter tensors.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.

    Returns:
        mean_pred (np.ndarray): Mean of sampled class probabilities, shape (N, num_classes).
        var_pred (np.ndarray): Variance of sampled class probabilities, shape (N, num_classes).
    """
    model.eval()
    device = next(model.parameters()).device  # Ensure data is on the correct device

    predictions = []

    for theta_sample in theta_samples:
        # Load new parameters into model
        torch.nn.utils.vector_to_parameters(theta_sample, model.parameters())
        model.to(device)
        sample_probs = []

        # Compute predictions (logits) and convert to probabilities using softmax
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = F.softmax(logits, dim=1)  # Convert logits to class probabilities
                sample_probs.append(probs.cpu())

        sample_probs = torch.cat(sample_probs, dim=0).numpy()
        predictions.append(sample_probs)

    predictions = np.stack(predictions, axis=0) # Shape: (num_samples, N, num_classes)

    # Compute mean and variance over the sampled weights
    mean_pred = np.mean(predictions, axis=0)  # Shape: (N, num_classes)
    var_pred = np.var(predictions, axis=0)  # Shape: (N, num_classes)

    return mean_pred, var_pred