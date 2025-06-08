from matplotlib import pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.ticker import LogFormatter

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

def plot_2D_decision_boundary_entropy(model, theta_samples, X_test, y_test, mean_probs_test, resolution=300, alpha=1.0, save_path="results/decision_boundaries/Bayesian_decision_boundary_entropy.png"):
    """
    Plots predictive entropy as the background and overlays test points with class-color and correctness markers.

    Args:
        model: Trained Bayesian model.
        theta_samples: Samples from q_LLA posterior (T x P).
        X_test (Tensor): Test inputs (N, 2)
        y_test (Tensor): Test labels (N,)
        resolution (int): Grid resolution.
        alpha (float): Background transparency.
    """
    device = next(model.parameters()).device

    # Create grid
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    # Predictive distribution over grid points
    mean_probs = []
    with torch.no_grad():
        for theta in theta_samples:
            torch.nn.utils.vector_to_parameters(theta, model.parameters())
            logits = model(grid_tensor)
            probs = torch.softmax(logits, dim=1)
            mean_probs.append(probs.cpu().numpy())

    mean_probs = np.stack(mean_probs, axis=0)  # (T, G, C)
    mean_pred = np.mean(mean_probs, axis=0)    # (G, C)

    # Predictive entropy
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-12), axis=1)
    entropy = entropy.reshape(xx.shape)
    entropy_max = np.log(mean_pred.shape[1])  # Max entropy

    # Plot
    num_classes = mean_pred.shape[1]
    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")

    plt.figure(figsize=(10, 6))

    # Plot entropy background
    entropy_img = plt.imshow(entropy, extent=(x_min, x_max, y_min, y_max), origin='lower',
                             cmap='inferno', alpha=alpha, vmin=0, vmax=entropy_max, aspect='auto')

    # Test point plotting
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test
    pred_test = np.argmax(mean_probs_test, axis=1)

    plotted_labels = set()
    for class_id in np.unique(y_test_np):
        mask = (y_test_np == class_id)
        correct = (pred_test == y_test_np) & mask
        wrong = (pred_test != y_test_np) & mask

        if correct.any():
            plt.scatter(X_test_np[correct, 0], X_test_np[correct, 1],
                        color=cmap(class_id % cmap.N), edgecolor='black', s=30, marker='o',
                        label=f"Class {class_id} ✓" if f"Class {class_id} ✓" not in plotted_labels else "")
            plotted_labels.add(f"Class {class_id} ✓")

        if wrong.any():
            plt.scatter(X_test_np[wrong, 0], X_test_np[wrong, 1],
                        color=cmap(class_id % cmap.N), edgecolor='black', s=50, marker='^',
                        label=f"Misclass {class_id}" if f"Misclass {class_id}" not in plotted_labels else "")
            plotted_labels.add(f"Misclass {class_id}")

    # Final polish
    plt.title("Bayesian Model: Predictive Entropy and Classification")
    plt.xlabel("X1")
    plt.ylabel("X2")
    # Place legend above the plot, outside to the left
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.2, 0.5),  # Push farther right (x = 1.25)
        borderaxespad=0.,
        title="Test Points"
    )
        # Place colorbar on right without overlap
    cbar = plt.colorbar(entropy_img, shrink=0.75, pad=0.02)
    cbar.set_label("Predictive Entropy")

    # Adjust layout so both legend and colorbar have space
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve 15% for right margin
    plt.savefig(save_path)
    plt.show()



def plot_2D_decision_boundary_confidence(model, theta_samples, X_test, y_test, mean_probs_test, resolution=300, alpha=1.0, save_path="results/decision_boundaries/Bayesian_decision_boundary_confidence.png"):
    """
    Plots a confidence map as a grayscale background where brightness increases near decision boundaries.
    Test points are colored by class and marked by correctness (circles/triangles).

    Args:
        model: Trained Bayesian model
        theta_samples: Posterior samples (T x P)
        X_test: Test input tensor (N x 2)
        y_test: Test label tensor (N,)
        resolution: Grid resolution for background
        alpha: Background transparency
    """
    device = next(model.parameters()).device

    # Build grid over input space
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    # Predictive mean probabilities over the grid
    probs_all = []
    with torch.no_grad():
        for theta in theta_samples:
            torch.nn.utils.vector_to_parameters(theta, model.parameters())
            logits = model(grid_tensor)
            probs = torch.softmax(logits, dim=1)
            probs_all.append(probs.cpu().numpy())

    probs_all = np.stack(probs_all, axis=0)  # (T, G, C)
    mean_probs = np.mean(probs_all, axis=0)  # (G, C)
    confidence = np.max(mean_probs, axis=1)  # (G,)
    confidence = confidence.reshape(xx.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot confidence background (inverted: high conf = dark, low conf = bright)
    confidence_img = ax.imshow(confidence, extent=(x_min, x_max, y_min, y_max), origin="lower",
                               cmap="BuPu", alpha=alpha, aspect="auto", vmin=0.0, vmax=1.0)

    # Colorbar
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = cm.ScalarMappable(cmap="BuPu", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Confidence")

    # Plot test points
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test
    pred_test = np.argmax(mean_probs_test, axis=1)
    num_classes = mean_probs.shape[1]
    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")

    for class_id in np.unique(y_test_np):
        mask = (y_test_np == class_id)
        correct = (pred_test == y_test_np) & mask
        wrong = (pred_test != y_test_np) & mask

        if correct.any():
            ax.scatter(X_test_np[correct, 0], X_test_np[correct, 1],
                       color=cmap(class_id % cmap.N), edgecolor='black',
                       s=30, marker='o', label=f"Class {class_id}")
        if wrong.any():
            ax.scatter(X_test_np[wrong, 0], X_test_np[wrong, 1],
                       color=cmap(class_id % cmap.N), edgecolor='black',
                       s=50, marker='^', label=f"Misclass {class_id}")

    # Final polish
    ax.set_title("Confidence Map: Bright = Uncertain, Dark = Confident")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.2, 0.5),
        borderaxespad=0.,
        title="Test Points"
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path)
    plt.show()

def plot_projection_metrics(proj_norms_list, kernel_ratios_list, sample_labels=None, path="results/metrics/alternating_projections/MNIST_metrics_plot.png"):
    """
    Plots projection norms and kernel norm ratios over iterations for multiple samples.

    Args:
        proj_norms_list (List[Tensor]): List of 1D tensors, each containing projection norms per iteration.
        kernel_ratios_list (List[Tensor]): List of 1D tensors, each containing kernel norm ratios per iteration.
        sample_labels (List[str], optional): Labels for each sample. Defaults to Sample 0, Sample 1, ...
    """
    import torch

    num_samples = len(proj_norms_list)
    iterations = [torch.arange(len(pn)).cpu().numpy() for pn in proj_norms_list]
    if sample_labels is None:
        sample_labels = [f"Sample {i}" for i in range(num_samples)]

    # Plot projection norms
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    for i in range(num_samples):
        ax1.plot(iterations[i], proj_norms_list[i].cpu().numpy(), label=sample_labels[i])
    ax1.set_title("Projection Norms over Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Projection Norm")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    ax1.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    for i in range(num_samples):
        ax2.plot(iterations[i], kernel_ratios_list[i].cpu().numpy(), label=sample_labels[i])
    ax2.set_title("Kernel Norm Ratios over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Kernel Norm Ratio")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    ax2.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.show()