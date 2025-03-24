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

def plot_2D_decision_boundary_MAP(model, test_dataset, resolution=300, alpha=0.6):
    """
    Plots decision boundaries and test points for a trained 2D classifier.

    - Correct test points: colored circles
    - Misclassified test points: colored triangles
    - Background: semi-transparent class regions
    - Bold black contours: decision boundaries

    Args:
        model (torch.nn.Module): Trained model.
        test_dataset (Dataset): Dataset with `.X` and `.y` attributes (tensors).
        resolution (int): Resolution of background grid.
        title (str): Title for the plot.
        alpha (float): Transparency of background class regions.
    """
    model.eval()
    device = next(model.parameters()).device

    # Create meshgrid for background
    x_min, x_max = test_dataset.X[:, 0].min() - 0.5, test_dataset.X[:, 0].max() + 0.5
    y_min, y_max = test_dataset.X[:, 1].min() - 0.5, test_dataset.X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(grid_tensor)
        grid_preds = torch.argmax(logits, dim=1).cpu().numpy()

    zz = grid_preds.reshape(xx.shape)

    # Test data
    X_test = test_dataset.X.numpy()
    y_test = test_dataset.y.numpy()
    with torch.no_grad():
        pred_test = torch.argmax(model(test_dataset.X.to(device)), dim=1).cpu().numpy()

    accuracy = (pred_test == y_test).mean() * 100

    num_classes = len(np.unique(y_test))
    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")

    # Plot decision regions
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, alpha=alpha, cmap=cmap, levels=np.arange(num_classes + 1) - 0.5)

    # Add bold decision boundaries
    plt.contour(xx, yy, zz, colors='black', linewidths=1.5, levels=np.arange(num_classes + 1) - 0.5)

    # Plot test points: triangles for misclassified, circles for correct
    for class_id in np.unique(y_test):
        class_mask = y_test == class_id
        correct_mask = (y_test == pred_test) & class_mask
        wrong_mask = (y_test != pred_test) & class_mask

        # Correct predictions → circles
        plt.scatter(X_test[correct_mask, 0], X_test[correct_mask, 1],
                    color=cmap(class_id % cmap.N),
                    edgecolor='black', linewidth=0.8, s=30, label=f"Class {class_id}", marker='o')

        # Misclassified predictions → triangles
        plt.scatter(X_test[wrong_mask, 0], X_test[wrong_mask, 1],
                    color=cmap(class_id % cmap.N),
                    edgecolor='black', linewidth=0.8, s=50, label=f"Misclassified {class_id}", marker='^')

    plt.title(f'Decision boundary (MAP) - Accuracy: {accuracy:.2f}%')
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.legend(
        loc="center left",       # anchor legend to the left of the bbox anchor
        bbox_to_anchor=(1.02, 0.5),  # move it slightly right of the plot, center-aligned
        borderaxespad=0.,
        title="Classes"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/decision_boundaries/MAP_decision_boundary.png')
    plt.show()
