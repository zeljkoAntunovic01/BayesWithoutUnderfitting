import json
import models.sinemodel as sinemodel
import models.mnistmodel as mnistmodel
from data import generate_sine_data, load_MNIST_data
from train import train_mnist, train_sine
import torch
from matplotlib import pyplot as plt
import numpy as np
from plots import plot_model, plot_bayesian_model_samples, plot_bayesian_model_predictions
import os
from utils import disassemble_data_loader, sample_from_posterior, save_metrics

SINE_MODEL_PATH="results/models/sine_net.pth"
MNIST_MODEL_PATH="results/models/mnist_fc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def lla_inference(model, x_train, y_train):
    theta_map, covariance = sinemodel.compute_q_lla(model, x_train, y_train)
    print("MAP Parameters:", theta_map)
    print("Posterior Covariance Shape:", covariance.shape)

    num_samples = 100
    theta_samples = sample_from_posterior(theta_map, covariance, num_samples)
    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()


    for i in range(len(theta_samples)):
        if i % 10 == 0:
            print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")

    x_test = torch.linspace(-5 * np.pi, 5 * np.pi, 200).unsqueeze(1)
    y_test_true = np.sin(x_test.numpy())  # True function values
    y_test_pred = model(x_test).detach().numpy()  # MAP model predictions
    mean_pred, var_pred = sinemodel.bayesian_prediction(model, theta_samples, x_test)

    metrics = save_metrics(y_test_pred, mean_pred, var_pred, y_test_true, "results/metrics/sine_lla_metrics.json")
    rmse_map = metrics["MAP"]["RMSE"]
    rmse_bayesian = metrics["Bayesian"]["RMSE"]

    plot_bayesian_model_samples(model, theta_samples, x_test, posterior="qLLA")
    plot_bayesian_model_predictions(x_train, y_train, x_test, y_test_true, y_test_pred, mean_pred, var_pred, rmse_map, rmse_bayesian, posterior="qLLA")

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def lla_inference_mnist(model, train_loader, test_loader):
    model.to(DEVICE)
    model.eval()

    x_train, y_train = disassemble_data_loader(train_loader)
    x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)

    # Compute q_LLA posterior
    theta_map, covariance = mnistmodel.compute_q_lla(model, x_train, y_train)
    num_samples = 100
    theta_samples = sample_from_posterior(theta_map, covariance, num_samples)
    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

    for i in range(len(theta_samples)):
        if i % 10 == 0:
            print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")


    x_test, y_test = disassemble_data_loader(test_loader)
    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)

    print("Performing MAP inference...")
    with torch.no_grad():
        logits_map = model(x_test)  # Standard model predictions (MAP)
        preds_map = logits_map.argmax(dim=1)  # Get class predictions
        acc_map = (preds_map == y_test).float().mean().item()  # Compute accuracy

    print(f"MAP Model Accuracy: {acc_map:.4f}")

    print("Performing Bayesian inference...")
    mean_pred, _ = mnistmodel.bayesian_prediction(model, theta_samples, x_test)  # Bayesian inference
    preds_bayesian = mean_pred.argmax(axis=1)  # Get class predictions
    acc_bayesian = (preds_bayesian == y_test.cpu().numpy()).mean()  # Compute accuracy

    print(f"Bayesian Model Accuracy: {acc_bayesian:.4f}")


    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def projected_posterior_inference(model, x_train, y_train):
    theta_map, covariance = sinemodel.compute_q_proj(model, x_train, y_train)
    print("MAP Parameters:", theta_map)
    print("Posterior Covariance Shape:", covariance.shape)

    num_samples = 100
    theta_samples = sample_from_posterior(theta_map, covariance, num_samples)
    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

    for i in range(len(theta_samples)):
        if i % 10 == 0:
            print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")

    x_test = torch.linspace(-5 * np.pi, 5 * np.pi, 200).unsqueeze(1)
    y_test_true = np.sin(x_test.numpy())  # True function values
    y_test_pred = model(x_test).detach().numpy()  # MAP model predictions
    mean_pred, var_pred = sinemodel.bayesian_prediction(model, theta_samples, x_test)

    metrics = save_metrics(y_test_pred, mean_pred, var_pred, y_test_true, "results/metrics/sine_projected_posterior_metrics.json")
    rmse_map = metrics["MAP"]["RMSE"]
    rmse_bayesian = metrics["Bayesian"]["RMSE"]

    plot_bayesian_model_samples(model, theta_samples, x_test, posterior="qPROJ")
    plot_bayesian_model_predictions(x_train, y_train, x_test, y_test_true, y_test_pred, mean_pred, var_pred, rmse_map, rmse_bayesian, posterior="qPROJ")

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def loss_posterior_inference(model, x_train, y_train):
    theta_map, covariance = sinemodel.compute_q_loss(model, x_train, y_train)
    print("MAP Parameters:", theta_map)
    print("Posterior Covariance Shape:", covariance.shape)

    num_samples = 100
    theta_samples = sample_from_posterior(theta_map, covariance, num_samples)
    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

    for i in range(len(theta_samples)):
        if i % 10 == 0:
            print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")

    x_test = torch.linspace(-5 * np.pi, 5 * np.pi, 200).unsqueeze(1)
    y_test_true = np.sin(x_test.numpy())  # True function values
    y_test_pred = model(x_test).detach().numpy()  # MAP model predictions
    mean_pred, var_pred = sinemodel.bayesian_prediction(model, theta_samples, x_test)

    metrics = save_metrics(y_test_pred, mean_pred, var_pred, y_test_true, "results/metrics/sine_loss_posterior_metrics.json")
    rmse_map = metrics["MAP"]["RMSE"]
    rmse_bayesian = metrics["Bayesian"]["RMSE"]

    plot_bayesian_model_samples(model, theta_samples, x_test, posterior="qLOSS")
    plot_bayesian_model_predictions(x_train, y_train, x_test, y_test_true, y_test_pred, mean_pred, var_pred, rmse_map, rmse_bayesian, posterior="qLOSS")

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def run_sine_experiment():
    # Train the MAP estimator
    x_train, y_train = generate_sine_data(1000)
    model = sinemodel.SineNet()
    if not os.path.exists(SINE_MODEL_PATH):
        train_sine(model=model, data=(x_train, y_train))
    else:
        model.load_state_dict(torch.load(SINE_MODEL_PATH))
    
    model.eval()
    plot_model(model=model, x_train=x_train, y_train=y_train)

    #lla_inference(model, x_train, y_train)
    #projected_posterior_inference(model, x_train, y_train)
    loss_posterior_inference(model, x_train, y_train)

def run_MNIST_experiment():
    train_data, test_data = load_MNIST_data()
    model = mnistmodel.MNIST_FC()
    if not os.path.exists(MNIST_MODEL_PATH):
        train_mnist(model=model, data=train_data, save_path=MNIST_MODEL_PATH)
    else:
        model.load_state_dict(torch.load(MNIST_MODEL_PATH))
    
    model.eval()
    lla_inference_mnist(model, train_data, test_data)

if __name__ == "__main__":
    run_sine_experiment()
    #run_MNIST_experiment()