import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


# Generate sine wave dataset
def generate_sine_data(n_samples=100, noise_std=0.1):
    x = torch.linspace(-2 * np.pi, 2 * np.pi, n_samples).unsqueeze(1)  # Shape: (n_samples, 1)
    y = torch.sin(x) + noise_std * torch.randn_like(x)  # Adding noise
    return x, y

# Load a subset of the MNIST dataset
def load_MNIST_data(train_size=10, test_size=2, batch_size=64):
    """
    Loads a subset of the MNIST dataset and returns DataLoader objects for training and testing.

    Args:
        train_size (int): Number of training samples to load.
        test_size (int): Number of test samples to load.
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
    ])

    # Load the full MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Select a subset of the dataset
    train_subset = Subset(train_dataset, range(train_size))
    test_subset = Subset(test_dataset, range(test_size))

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
