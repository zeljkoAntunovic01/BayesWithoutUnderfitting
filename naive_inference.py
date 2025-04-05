import models.sinemodel as sinemodel
import models.fcmodel as fcmodel
import torch
import numpy as np
from plots import plot_2D_decision_boundary_confidence, plot_2D_decision_boundary_entropy, plot_bayesian_model_samples, plot_bayesian_model_predictions
from utils import sample_from_posterior, save_metrics_classification, save_metrics_regression

DECISION_BOUNDARIES_PATH = "results/decision_boundaries/naive_approach/"
METRICS_PATH = "results/metrics/naive_approach/"

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

    metrics = save_metrics_regression(y_test_pred, mean_pred, var_pred, y_test_true, f"{METRICS_PATH}sine_lla_metrics.json")
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
    y_test_pred_map = torch.nn.functional.softmax(model(x_test), dim=1).detach().numpy()
    mean_probs_pred, var_pred = fcmodel.bayesian_prediction(model, theta_samples, x_test)
    y_test_pred = np.argmax(mean_probs_pred, axis=1)
    
    metrics = save_metrics_classification(y_test_pred_map, y_test_pred, mean_probs_pred, var_pred, y_test_true.detach().numpy(), f"{METRICS_PATH}2Dclassifier_lla_metrics.json")

    plot_2D_decision_boundary_entropy(model, theta_samples, x_test, y_test_true, mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_lla_entropy.png")
    plot_2D_decision_boundary_confidence(model, theta_samples, x_test, y_test_true, mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_lla_confidence.png")

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

    metrics = save_metrics_regression(y_test_pred, mean_pred, var_pred, y_test_true, f"{METRICS_PATH}sine_projected_posterior_metrics.json")
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
    y_test_pred_map = torch.nn.functional.softmax(model(x_test), dim=1).detach().numpy()
    mean_probs_pred, var_pred = fcmodel.bayesian_prediction(model, theta_samples, x_test)
    y_test_pred = np.argmax(mean_probs_pred, axis=1)
    
    metrics = save_metrics_classification(y_test_pred_map, y_test_pred, mean_probs_pred, var_pred, y_test_true.detach().numpy(), f"{METRICS_PATH}2Dclassifier_proj_metrics.json")

    plot_2D_decision_boundary_entropy(model, theta_samples, x_test, y_test_true, mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_proj_entropy.png")
    plot_2D_decision_boundary_confidence(model, theta_samples, x_test, y_test_true, mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_proj_confidence.png")

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

    metrics = save_metrics_regression(y_test_pred, mean_pred, var_pred, y_test_true, f"{METRICS_PATH}sine_loss_posterior_metrics.json")
    rmse_map = metrics["MAP"]["RMSE"]
    rmse_bayesian = metrics["Bayesian"]["RMSE"]

    plot_bayesian_model_samples(model, theta_samples, x_test, posterior="qLOSS")
    plot_bayesian_model_predictions(x_train, y_train, x_test, y_test_true, y_test_pred, mean_pred, var_pred, rmse_map, rmse_bayesian, posterior="qLOSS")

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def loss_posterior_inference_2D_classifier(model, train_dataset, test_dataset):
    x_train = train_dataset[:][0]
    y_train = train_dataset[:][1]
    theta_map, covariance = fcmodel.compute_q_loss(model, x_train, y_train)
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
    y_test_pred_map = torch.nn.functional.softmax(model(x_test), dim=1).detach().numpy()
    mean_probs_pred, var_pred = fcmodel.bayesian_prediction(model, theta_samples, x_test)
    y_test_pred = np.argmax(mean_probs_pred, axis=1)
    
    metrics = save_metrics_classification(y_test_pred_map, y_test_pred, mean_probs_pred, var_pred, y_test_true.detach().numpy(), f"{METRICS_PATH}2Dclassifier_loss_metrics.json")

    plot_2D_decision_boundary_entropy(model, theta_samples, x_test, y_test_true, mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_loss_entropy.png")
    plot_2D_decision_boundary_confidence(model, theta_samples, x_test, y_test_true, mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_loss_confidence.png")

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()  
