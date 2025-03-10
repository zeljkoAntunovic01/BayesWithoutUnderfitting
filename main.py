from model import SineNet, compute_q_lla, sample_from_posterior, bayesian_prediction, compute_q_proj, compute_q_loss
from data import generate_sine_data
from train import train
import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from plots import plot_model, plot_bayesian_model_samples, plot_bayesian_model_predictions
import os

MODEL_PATH="models/sine_net.pth"

def lla_inference(model, x_train, y_train):
    theta_map, covariance = compute_q_lla(model, x_train, y_train)
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
    mean_pred, var_pred = bayesian_prediction(model, theta_samples, x_test)

    rmse_map = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    rmse_bayesian = np.sqrt(mean_squared_error(y_test_true, mean_pred))

    print(f"RMSE (MAP Model): {rmse_map:.4f}")
    print(f"RMSE (Bayesian Mean Model): {rmse_bayesian:.4f}")

    plot_bayesian_model_samples(model, theta_samples, x_test, posterior="qLLA")
    plot_bayesian_model_predictions(x_train, y_train, x_test, y_test_true, y_test_pred, mean_pred, var_pred, rmse_map, rmse_bayesian, posterior="qLLA")

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def projected_posterior_inference(model, x_train, y_train):
    theta_map, covariance = compute_q_proj(model, x_train, y_train)
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
    mean_pred, var_pred = bayesian_prediction(model, theta_samples, x_test)

    rmse_map = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    rmse_bayesian = np.sqrt(mean_squared_error(y_test_true, mean_pred))

    print(f"RMSE (MAP Model): {rmse_map:.4f}")
    print(f"RMSE (Bayesian Mean Model): {rmse_bayesian:.4f}")

    plot_bayesian_model_samples(model, theta_samples, x_test, posterior="qPROJ")
    plot_bayesian_model_predictions(x_train, y_train, x_test, y_test_true, y_test_pred, mean_pred, var_pred, rmse_map, rmse_bayesian, posterior="qPROJ")

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def loss_posterior_inference(model, x_train, y_train):
    theta_map, covariance = compute_q_loss(model, x_train, y_train)
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
    mean_pred, var_pred = bayesian_prediction(model, theta_samples, x_test)

    rmse_map = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    rmse_bayesian = np.sqrt(mean_squared_error(y_test_true, mean_pred))

    print(f"RMSE (MAP Model): {rmse_map:.4f}")
    print(f"RMSE (Bayesian Mean Model): {rmse_bayesian:.4f}")

    plot_bayesian_model_samples(model, theta_samples, x_test, posterior="qLOSS")
    plot_bayesian_model_predictions(x_train, y_train, x_test, y_test_true, y_test_pred, mean_pred, var_pred, rmse_map, rmse_bayesian, posterior="qLOSS")

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

if __name__ == "__main__":
    # Train the MAP estimator
    x_train, y_train = generate_sine_data(1000)
    model = SineNet()
    if not os.path.exists(MODEL_PATH):
        train(model=model, data=(x_train, y_train))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
    
    model.eval()
    plot_model(model=model, x_train=x_train, y_train=y_train)

    lla_inference(model, x_train, y_train)
    projected_posterior_inference(model, x_train, y_train)
    loss_posterior_inference(model, x_train, y_train)