
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import cifar10model, fcmodel, mnistmodel
from plots import plot_2D_decision_boundary_confidence, plot_2D_decision_boundary_entropy
from utils import alternating_projections_qloss_classifier, alternating_projections_qproj_classifier, save_metrics_classification

DECISION_BOUNDARIES_PATH = "results/decision_boundaries/alternating_projections/"
METRICS_PATH = "results/metrics/alternating_projections/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def proj_posterior_inference_2D_classifier_alt(model, train_dataset, test_dataset):
    x_train = train_dataset[:][0]
    
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    num_samples = 5
    theta_samples = alternating_projections_qproj_classifier(model, x_train, alpha=10.0, num_samples=num_samples)

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
    
    metrics = save_metrics_classification(
        y_test_pred_map, y_test_pred, mean_probs_pred, var_pred,
        y_test_true.detach().numpy(), f"{METRICS_PATH}2Dclassifier_proj_alt_metrics.json"
    )

    plot_2D_decision_boundary_entropy(
        model, theta_samples, x_test, y_test_true,
        mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_proj_alt_entropy.png"
    )

    plot_2D_decision_boundary_confidence(
        model, theta_samples, x_test, y_test_true,
        mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_proj_alt_confidence.png"
    )

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def loss_posterior_inference_2D_classifier_alt(model, train_dataset, test_dataset):
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    num_samples = 5
    theta_samples = alternating_projections_qloss_classifier(model, train_dataset, alpha=10.0, num_samples=num_samples)

    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

    for i in range(len(theta_samples)):
        #if i % 10 == 0:
        print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")
    
    x_test = test_dataset[:][0]
    y_test_true = test_dataset[:][1]
    y_test_pred_map = torch.nn.functional.softmax(model(x_test), dim=1).detach().numpy()
    mean_probs_pred, var_pred = fcmodel.bayesian_prediction(model, theta_samples, x_test)
    
    metrics = save_metrics_classification(
        y_test_pred_map, mean_probs_pred,
        y_test_true.detach().numpy(), f"{METRICS_PATH}2Dclassifier_loss_alt_metrics.json"
    )

    plot_2D_decision_boundary_entropy(
        model, theta_samples, x_test, y_test_true,
        mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_loss_alt_entropy.png"
    )

    plot_2D_decision_boundary_confidence(
        model, theta_samples, x_test, y_test_true,
        mean_probs_pred, save_path=f"{DECISION_BOUNDARIES_PATH}Decision_boundary_loss_alt_confidence.png"
    )

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def loss_posterior_inference_MNIST_alt(model, train_dataset, test_dataset):
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    num_samples = 5
    theta_samples = alternating_projections_qloss_classifier(model, train_dataset, alpha=30.0, num_samples=num_samples)

    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

    for i in range(len(theta_samples)):
        print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Get MAP predictions
    model.to(DEVICE)
    model.eval()
    y_test_pred_map_all = []
    y_test_true_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.nn.functional.softmax(logits, dim=1)
            y_test_pred_map_all.append(probs.cpu())
            y_test_true_all.append(yb.cpu())

    y_test_pred_map = torch.cat(y_test_pred_map_all, dim=0).numpy()
    y_test_true = torch.cat(y_test_true_all, dim=0).numpy()

    # Bayesian prediction (Mean, Variance)
    mean_probs_pred, var_pred = mnistmodel.bayesian_prediction(model, theta_samples, test_loader)
    
    # Save metrics
    metrics = save_metrics_classification(
        y_test_pred_map, mean_probs_pred,
        y_test_true, f"{METRICS_PATH}MNIST_loss_alt_metrics.json"
    )

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()

def loss_posterior_inference_CIFAR10_alt(model, train_dataset, test_dataset):
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    num_samples = 5
    theta_samples = alternating_projections_qloss_classifier(model, train_dataset, alpha=30.0, num_samples=num_samples)

    # Save original parameters
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

    for i in range(len(theta_samples)):
        print(f"Sample {i} Difference from MAP: {torch.norm(theta_samples[i] - theta_map).item()}")
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Get MAP predictions
    model.to(DEVICE)
    model.eval()
    y_test_pred_map_all = []
    y_test_true_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.nn.functional.softmax(logits, dim=1)
            y_test_pred_map_all.append(probs.cpu())
            y_test_true_all.append(yb.cpu())

    y_test_pred_map = torch.cat(y_test_pred_map_all, dim=0).numpy()
    y_test_true = torch.cat(y_test_true_all, dim=0).numpy()

    # Bayesian prediction (Mean, Variance)
    mean_probs_pred, var_pred = cifar10model.bayesian_prediction(model, theta_samples, test_loader)
    
    # Save metrics
    metrics = save_metrics_classification(
        y_test_pred_map, mean_probs_pred,
        y_test_true, f"{METRICS_PATH}CIFAR10_loss_alt_metrics.json"
    )

    # Restore original MAP parameters
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())
    model.eval()