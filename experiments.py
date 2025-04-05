import models.sinemodel as sinemodel
import models.fcmodel as fcmodel
from data.utils import generate_sine_data
from data.gaussian_2d_dataset import Gaussian2DClassificationDataset
from naivesampling import lla_inference, lla_inference_2D_classifier, loss_posterior_inference, loss_posterior_inference_2D_classifier, projected_posterior_inference, projected_posterior_inference_2D_classifier
from train import train_classifier, train_sine
import torch
from torch.utils.data import DataLoader
from plots import plot_2D_decision_boundary_MAP, plot_model
import os

SINE_MODEL_PATH="results/models/sine_net.pth"
MNIST_MODEL_PATH="results/models/mnist_fc.pth"
FC_2D_MODEL_PATH="results/models/fc_2d_net.pth"
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
        train_classifier(model=model, data=train_loader, save_path=FC_2D_MODEL_PATH)
    else:
        model.load_state_dict(torch.load(FC_2D_MODEL_PATH))
    
    model.eval()
    plot_2D_decision_boundary_MAP(model, test_dataset)

    lla_inference_2D_classifier(model, train_dataset, test_dataset)
    projected_posterior_inference_2D_classifier(model, train_dataset, test_dataset)
    loss_posterior_inference_2D_classifier(model, train_dataset, test_dataset)