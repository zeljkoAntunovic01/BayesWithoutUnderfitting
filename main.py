import models.sinemodel as sinemodel
import models.fcmodel as fcmodel
from data.utils import generate_sine_data
from data.gaussian_2d_dataset import Gaussian2DClassificationDataset
from train import train_classifier, train_sine
import torch
from torch.utils.data import DataLoader
import numpy as np
from plots import plot_2D_decision_boundary_MAP, plot_2D_decision_boundary_confidence, plot_2D_decision_boundary_entropy, plot_model, plot_bayesian_model_samples, plot_bayesian_model_predictions
import os
from utils import sample_from_posterior, save_metrics

SINE_MODEL_PATH="results/models/sine_net.pth"
MNIST_MODEL_PATH="results/models/mnist_fc.pth"
FC_2D_MODEL_PATH="results/models/fc_2d_net.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def lla_inference(model, x_train, y_train):
    theta_map, covariance = sinemodel.compute_q_lla(model, x_train, y_train)

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

def lla_inference_2D_classifier(model, train_dataset, test_dataset):
    x_train = train_dataset[:][0]
    y_train = train_dataset[:][1]
    theta_map, covariance = fcmodel.compute_q_lla(model, x_train, y_train)
    print("MAP Parameters shape:", theta_map.shape)
    print("Posterior Covariance Shape:", covariance.shape)

    num_samples = 100
    theta_samples = sample_from_posterior(theta_map, covariance, num_samples)

    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

    for i in range(len(theta_samples)):
        if i % 10 == 0:
            print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")
    
    x_test = test_dataset[:][0]
    y_test_true = test_dataset[:][1]
    mean_pred, var_pred = fcmodel.bayesian_prediction(model, theta_samples, x_test)

    plot_2D_decision_boundary_entropy(model, theta_samples, x_test, y_test_true, mean_pred)
    plot_2D_decision_boundary_confidence(model, theta_samples, x_test, y_test_true, mean_pred)

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

def projected_posterior_inference_2D_classifier(model, train_dataset, test_dataset):
    x_train = train_dataset[:][0]
    y_train = train_dataset[:][1]
    theta_map, covariance = fcmodel.compute_q_proj(model, x_train, y_train)
    print("MAP Parameters shape:", theta_map.shape)
    print("Posterior Covariance Shape:", covariance.shape)

    num_samples = 100
    theta_samples = sample_from_posterior(theta_map, covariance, num_samples)

    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

    for i in range(len(theta_samples)):
        if i % 10 == 0:
            print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")
    
    x_test = test_dataset[:][0]
    y_test_true = test_dataset[:][1]
    mean_pred, var_pred = fcmodel.bayesian_prediction(model, theta_samples, x_test)

    plot_2D_decision_boundary_entropy(model, theta_samples, x_test, y_test_true, mean_pred)
    plot_2D_decision_boundary_confidence(model, theta_samples, x_test, y_test_true, mean_pred)

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

    lla_inference(model, x_train, y_train)
    #projected_posterior_inference(model, x_train, y_train)
    #loss_posterior_inference(model, x_train, y_train)

def run_2d_classification_experiment():
    train_dataset = Gaussian2DClassificationDataset(split="train", n_classes=4, points_per_class=100)
    test_dataset = Gaussian2DClassificationDataset(split="test", n_classes=4, points_per_class=100)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = fcmodel.FC_2D_Net(n_classes=4)
    if not os.path.exists(FC_2D_MODEL_PATH):
        train_classifier(model=model, data=train_loader, save_path=FC_2D_MODEL_PATH)
    else:
        model.load_state_dict(torch.load(FC_2D_MODEL_PATH))
    
    model.eval()
    #plot_2D_decision_boundary_MAP(model, test_dataset)

    #lla_inference_2D_classifier(model, train_dataset, test_dataset)
    projected_posterior_inference_2D_classifier(model, train_dataset, test_dataset)


if __name__ == "__main__":
    #run_sine_experiment()
    run_2d_classification_experiment()