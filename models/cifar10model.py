import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights

class CIFAR10_Net(nn.Module):
    def __init__(self, fine_tuning: bool = True, num_classes: int = 10):
        """
        Initializes the pre-trained ResNet18Model with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
        """
        super(CIFAR10_Net, self).__init__()

        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)

        # Modify for CIFAR-10 input size (32x32)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove 2x2 maxpool

        if fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

def bayesian_prediction(model, theta_samples, test_loader):
    """
    Makes Bayesian predictions by averaging over posterior samples for a classification model (MNIST).

    Args:
        model (torch.nn.Module): Neural network model (e.g., CIFAR10_Net).
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