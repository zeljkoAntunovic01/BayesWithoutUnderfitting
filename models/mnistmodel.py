import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.convolutional = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.MaxPool2d(kernel_size=(2,2)),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.1))

        self.fully_connected = nn.Sequential(
                nn.Linear(14*14*16, 500),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(500, 10))

    def forward(self, x):
        x = self.convolutional(x)
        # Reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
    
def bayesian_prediction(model, theta_samples, test_loader):
    """
    Makes Bayesian predictions by averaging over posterior samples for a classification model (MNIST).

    Args:
        model (torch.nn.Module): Neural network model (e.g., FC_2D_Net).
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