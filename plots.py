from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_model(model, x_train, y_train):
    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 200).unsqueeze(1)
    model.eval()
    y_test_pred = model(x_test).detach()
    plt.scatter(x_train.numpy(), y_train.numpy(), label="Training Data", color="blue")
    plt.plot(x_test.numpy(), np.sin(x_test.numpy()), label="True Function", color="red", linestyle="dashed")
    plt.plot(x_test.numpy(), y_test_pred.numpy(), label="Model Prediction (MAP)", color="yellow")
    plt.legend()
    plt.savefig("results/sine_wave.png")

def plot_bayesian_model_samples(model, theta_samples, x_test, posterior="qLLA"):
    plt.clf()
    for i, theta_sample in enumerate(theta_samples[:5]):  # Show 5 sample predictions
        torch.nn.utils.vector_to_parameters(theta_sample, model.parameters())
        y_sample = model(x_test).detach().numpy()
        plt.plot(x_test.numpy(), y_sample, alpha=0.5, label=f"Sample {i}", linestyle='--')

    plt.legend()
    plt.savefig(f'results/bayesian_samples_{posterior}.png')

def plot_bayesian_model_predictions(x_train, y_train, x_test, y_test_true, y_test_pred, mean_pred, var_pred, rmse_map, rmse_bayesian, posterior="qLLA"):
    plt.clf()
    plt.scatter(x_train.numpy(), y_train.numpy(), label="Training Data", color="blue")
    plt.plot(x_test.numpy(), y_test_true, label="True Function", color="red", linestyle="dashed")
    plt.plot(x_test.numpy(), mean_pred, label="Mean Prediction (Bayesian)", color="green")
    plt.plot(x_test.numpy(), y_test_pred, label="Model Prediction (MAP)", color="yellow")
    plt.fill_between(
        x_test.numpy().squeeze(),
        (mean_pred - 2 * np.sqrt(var_pred)).squeeze(),
        (mean_pred + 2 * np.sqrt(var_pred)).squeeze(),
        color="brown",
        alpha=0.3,
        label="Uncertainty (2 std)",
    )

    # Add accuracy metrics to the plot
    plt.text(
        0.05, 0.95,
        f"RMSE (MAP): {rmse_map:.4f}\nRMSE (Bayesian): {rmse_bayesian:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.legend(loc="lower right")
    plt.savefig(f'results/bayesian_sine_wave_{posterior}.png')
    plt.show()