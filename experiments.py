import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.cifar10model import CIFAR10_Net
from models.mnistmodel import MNIST_Net
import models.sinemodel as sinemodel
import models.fcmodel as fcmodel
from data.utils import generate_sine_data
from data.gaussian_2d_dataset import Gaussian2DClassificationDataset
from naive_inference import lla_inference, lla_inference_2D_classifier, loss_posterior_inference, loss_posterior_inference_2D_classifier, projected_posterior_inference, projected_posterior_inference_2D_classifier
from train import train_classifier, train_sine
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from plots import plot_2D_decision_boundary_MAP, plot_model
import os
from alternating_projections_inference import loss_posterior_inference_2D_classifier_alt, loss_posterior_inference_CIFAR10_alt, loss_posterior_inference_MNIST_alt, proj_posterior_inference_2D_classifier_alt

SINE_MODEL_PATH="results/models/sine_net.pth"
MNIST_MODEL_PATH="results/models/mnist_model.pth"
FC_2D_MODEL_PATH="results/models/fc_2d_net.pth"
CIFAR10_MODEL_PATH="results/models/cifar10_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_naive_sine_experiment():
    # Train the MAP estimator
    x_train, y_train = generate_sine_data(1000)
    model = sinemodel.SineNet()
    if not os.path.exists(SINE_MODEL_PATH):
        train_sine(model=model, data=(x_train, y_train))
    else:
        model.load_state_dict(torch.load(SINE_MODEL_PATH))
    
    model.eval()
    plot_model(model=model, x_train=x_train, y_train=y_train)

    lla_inference(model, x_train, y_train)
    projected_posterior_inference(model, x_train, y_train)
    loss_posterior_inference(model, x_train, y_train)

def run_naive_2d_classification_experiment():
    train_dataset = Gaussian2DClassificationDataset(split="train", n_classes=4, points_per_class=1000)
    test_dataset = Gaussian2DClassificationDataset(split="test", n_classes=4, points_per_class=100)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = fcmodel.FC_2D_Net(n_classes=4)
    if not os.path.exists(FC_2D_MODEL_PATH):
        train_classifier(model=model, train_data=train_loader, save_path=FC_2D_MODEL_PATH)
    else:
        model.load_state_dict(torch.load(FC_2D_MODEL_PATH))
    
    model.eval()
    plot_2D_decision_boundary_MAP(model, test_dataset)

    lla_inference_2D_classifier(model, train_dataset, test_dataset)
    projected_posterior_inference_2D_classifier(model, train_dataset, test_dataset)
    loss_posterior_inference_2D_classifier(model, train_dataset, test_dataset)

def run_alternated_projections_2d_classification_experiment():
    train_dataset = Gaussian2DClassificationDataset(split="train", n_classes=4, points_per_class=1000)
    test_dataset = Gaussian2DClassificationDataset(split="test", n_classes=4, points_per_class=100)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = fcmodel.FC_2D_Net(n_classes=4)
    if not os.path.exists(FC_2D_MODEL_PATH):
        train_classifier(model=model, train_data=train_loader, save_path=FC_2D_MODEL_PATH)
    else:
        model.load_state_dict(torch.load(FC_2D_MODEL_PATH))
    
    model.eval()
    #plot_2D_decision_boundary_MAP(model, test_dataset)

    #proj_posterior_inference_2D_classifier_alt(model, train_dataset, test_dataset)
    loss_posterior_inference_2D_classifier_alt(model, train_dataset, test_dataset)

def run_alternated_projections_MNIST_experiment(val_split=1.0/6.0):
    random.seed(42)
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),     # Slight translation
        transforms.RandomRotation(degrees=10),    # Small rotation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # FOR ACTUAL RUN
    """     # Load full dataset once with *no* transform
    full_train = datasets.MNIST(root='raw_data', train=True, download=True)

    # Create index split
    val_size = int(val_split * len(full_train))
    train_size = len(full_train) - val_size
    train_indices, val_indices = random_split(range(len(full_train)), [train_size, val_size])

    # Apply transforms to subsets via Subset and transform overrides
    train_dataset = Subset(datasets.MNIST(root='raw_data', train=True, transform=train_transform), train_indices)
    val_dataset = Subset(datasets.MNIST(root='raw_data', train=True, transform=test_transform), val_indices)
    test_dataset = datasets.MNIST(root='raw_data', train=False, download=True, transform=test_transform) """


    # FOR TESTING RUN:
    # Load full dataset without transform just to split
    full_train = datasets.MNIST(root='raw_data', train=True, download=True)
    full_test = datasets.MNIST(root='raw_data', train=False, download=True)

    # Get fixed number of samples
    train_indices = random.sample(range(len(full_train)), 100)
    val_indices = random.sample(list(set(range(len(full_train))) - set(train_indices)), 4)
    test_indices = random.sample(range(len(full_test)), 4)

    # Wrap subsets with transforms
    train_dataset = Subset(
        datasets.MNIST(root='raw_data', train=True, transform=train_transform),
        train_indices
    )
    val_dataset = Subset(
        datasets.MNIST(root='raw_data', train=True, transform=test_transform),
        val_indices
    )
    test_dataset = Subset(
        datasets.MNIST(root='raw_data', train=False, transform=test_transform),
        test_indices
    )


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MNIST_Net()
    if not os.path.exists(MNIST_MODEL_PATH):
        train_classifier(model=model, train_data=train_loader, val_data=val_loader, save_path=MNIST_MODEL_PATH)
    else:
        model.load_state_dict(torch.load(MNIST_MODEL_PATH))
    
    model.eval()

    loss_posterior_inference_MNIST_alt(model, train_dataset, test_dataset)

def run_alternated_projections_CIFAR10_experiment(val_split=1.0/6.0):
    random.seed(42)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # Maintain 32x32 size with padding
        transforms.RandomHorizontalFlip(),          # Basic augmentation
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


    # FOR ACTUAL RUN
    full_train = datasets.CIFAR10(root='raw_data', train=True, download=True)
    # Create index split
    val_size = int(val_split * len(full_train))
    train_size = len(full_train) - val_size
    train_indices, val_indices = random_split(range(len(full_train)), [train_size, val_size])
    # Apply transforms to subsets via Subset and transform overrides
    train_dataset = Subset(datasets.CIFAR10(root='raw_data', train=True, transform=train_transform), train_indices)
    val_dataset = Subset(datasets.CIFAR10(root='raw_data', train=True, transform=test_transform), val_indices)
    test_dataset = datasets.CIFAR10(root='raw_data', train=False, download=True, transform=test_transform)

    """ # FOR TESTING RUN:
    # Load full dataset without transform just to split
    full_train = datasets.CIFAR10(root='raw_data', train=True, download=True)
    full_test = datasets.CIFAR10(root='raw_data', train=False, download=True)
    # Get fixed number of samples
    train_indices = random.sample(range(len(full_train)), 1000)
    val_indices = random.sample(list(set(range(len(full_train))) - set(train_indices)), 400)
    test_indices = random.sample(range(len(full_test)), 400)
    # Wrap subsets with transforms
    train_dataset = Subset(
        datasets.CIFAR10(root='raw_data', train=True, transform=train_transform),
        train_indices
    )
    val_dataset = Subset(
        datasets.CIFAR10(root='raw_data', train=True, transform=test_transform),
        val_indices
    )
    test_dataset = Subset(
        datasets.CIFAR10(root='raw_data', train=False, transform=test_transform),
        test_indices
    ) """


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CIFAR10_Net(fine_tuning=False)
    if not os.path.exists(CIFAR10_MODEL_PATH):
        train_classifier(model=model, train_data=train_loader, val_data=val_loader, save_path=CIFAR10_MODEL_PATH)
    else:
        model.load_state_dict(torch.load(CIFAR10_MODEL_PATH))
    
    model.eval()

    loss_posterior_inference_CIFAR10_alt(model, train_dataset, test_dataset)