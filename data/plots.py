import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from utils import generate_sine_data, load_MNIST_data
from gaussian_2d_dataset import Gaussian2DClassificationDataset

def plot_sine_data():
    # Generate and visualize the data
    x_train, y_train = generate_sine_data(100)
    plt.scatter(x_train.numpy(), y_train.numpy(), label="Training Data")
    plt.plot(x_train.numpy(), np.sin(x_train.numpy()), color="red", label="True Function")
    plt.legend()
    plt.show()

def plot_MNIST_data():
    # Load the MNIST dataset
    train_loader, test_loader = load_MNIST_data(train_size=10000, test_size=2000)

    # Visualize a batch of images
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images, nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

def plot_multiclass_2d(n_classes=4):
    """
    Plots training and test splits from a PyTorch-style 2D classification dataset.

    Args:
        dataset_cls: A PyTorch Dataset class (e.g., Gaussian2DClassificationDataset)
        n_classes (int): Number of classes to help with colormap.
    """
    # Instantiate both splits
    train_dataset = Gaussian2DClassificationDataset(split="train", n_classes=n_classes)
    test_dataset = Gaussian2DClassificationDataset(split="test", n_classes=n_classes)

    # Extract tensors and convert to numpy
    X_train = train_dataset.X.numpy()
    y_train = train_dataset.y.numpy()
    X_test = test_dataset.X.numpy()
    y_test = test_dataset.y.numpy()

    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")

    # --- Plot training data ---
    plt.figure(figsize=(6, 6))
    for class_id in np.unique(y_train):
        mask = y_train == class_id
        plt.scatter(X_train[mask, 0], X_train[mask, 1],
                    label=f"Class {class_id}",
                    alpha=0.8,
                    s=20,
                    color=cmap(class_id % cmap.N))
    plt.title("Training Data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot test data ---
    plt.figure(figsize=(6, 6))
    for class_id in np.unique(y_test):
        mask = y_test == class_id
        plt.scatter(X_test[mask, 0], X_test[mask, 1],
                    label=f"Class {class_id}",
                    alpha=0.8,
                    s=20,
                    color=cmap(class_id % cmap.N))
    plt.title("Test Data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #plot_sine_data()
    #plot_MNIST_data()
    plot_multiclass_2d()
