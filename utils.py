import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from metrics import (
    compute_accuracy, compute_brier_score, compute_classification_ece_mce, compute_classification_nll, compute_confidence, compute_coverage, compute_multiclass_auroc, compute_predictive_entropy, compute_regression_ece, compute_regression_nll, compute_rmse, compute_sharpness
)
from torch.func import vmap, jvp, vjp, functional_call
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def compute_model_jacobian_params(model, x_train):
    """
    Computes the Jacobian of the model outputs w.r.t. model parameters manually using torch.autograd.grad().

    Args:
        model: PyTorch model (e.g., SineNet).
        x_train: Input data (torch.Tensor), shape (N, input_dim).

    Returns:
        J: Jacobian matrix (output_dim, num_params), shape (N, P).
    """
    model.eval()
    output = model(x_train)
    output = output.view(-1)  # Convert to shape (N,)

    num_params = sum(p.numel() for p in model.parameters())
    J = torch.zeros(output.shape[0], num_params)  # Shape: (N, P)

    for i in range(output.shape[0]):
        grad_output = torch.zeros_like(output)
        grad_output[i] = 1.0  # Select the i-th element to differentiate

        # Compute gradients with respect to model parameters
        grads = torch.autograd.grad(outputs=output, inputs=model.parameters(),
                                    grad_outputs=grad_output, retain_graph=True, allow_unused=True)

        # Flatten and concatenate gradients into the Jacobian
        grads = [g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, model.parameters())]
        J[i, :] = torch.cat(grads)

    return J

def compute_loss_jacobian_params(model, x_train, y_train, criterion):
    """
    Computes the Jacobian of the loss w.r.t. model parameters manually using torch.autograd.grad().

    Args:
        model: PyTorch model (e.g., SineNet).
        x_train: Input data (torch.Tensor), shape (N, input_dim).
        y_train: Target data (torch.Tensor), shape (N, output_dim).
        criterion: Loss function (e.g., nn.MSELoss()).	

    Returns:
        J: Jacobian matrix (output_dim, num_params), shape (N, P).
    """
    J_L_theta = []
    
    for x, y in zip(x_train, y_train):
        x = x.unsqueeze(0)  # Add batch dimension
        y = y.unsqueeze(0)
        
        model.zero_grad()  # Clear previous gradients
        y_pred = model(x) 
        loss = criterion(y_pred, y)
        
        # Compute gradients of the loss w.r.t. model parameters
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
        
        # Flatten and accumulate gradients
        J_L_theta.append(torch.cat([g.view(-1) for g in grads]))
    
    J_L_theta = torch.stack(J_L_theta)  # Shape: (N, P)
    return J_L_theta

def estimate_sigma_sine(model, x_train, y_train):
    """Estimate sigma^2 from dataset using residual variance."""
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train)
        residuals = y_train - y_pred
        sigma_sq = torch.mean(residuals**2)  # Estimate variance
        sigma = torch.sqrt(sigma_sq)  # Estimate standard deviation
    return sigma

def compute_model_jacobian_params_classifier(model, x_train):
    """
    Computes the Jacobian of the model outputs (logits) w.r.t. model parameters
    manually using torch.autograd.grad() for a classification model like MNIST CNN.

    Args:
        model: PyTorch classification model (e.g., MNIST_CNN).
        x_train: Input data (torch.Tensor), shape (N, 1, 28, 28).

    Returns:
        J: Jacobian matrix (N * num_classes, P), where:
           - N = number of samples
           - num_classes = number of output classes (10 for MNIST)
           - P = total number of model parameters
    """
    model.eval()
    
    # Move input to the same device as the model
    device = next(model.parameters()).device
    x_train = x_train.to(device)

    # Forward pass to get logits
    output = model(x_train)  # Shape: (N, num_classes)
    
    num_classes = output.shape[1]  # Should be 10 for MNIST
    num_samples = output.shape[0]  # N samples

    # Total number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())

    # Jacobian matrix storage
    J = torch.zeros(num_samples * num_classes, num_params, device=device)  # Shape: (N * 10, P)

    for i in range(num_samples):
        for j in range(num_classes):
            grad_output = torch.zeros_like(output)
            grad_output[i, j] = 1.0  # One-hot selection of the (i, j)th output

            # Compute gradient of logits[j] w.r.t. all model parameters
            grads = torch.autograd.grad(outputs=output, inputs=model.parameters(),
                                        grad_outputs=grad_output, retain_graph=True, allow_unused=True)

            # Flatten and concatenate gradients into the Jacobian
            grads = [g.view(-1) if g is not None else torch.zeros(p.numel(), device=device)
                     for g, p in zip(grads, model.parameters())]
            
            J[i * num_classes + j, :] = torch.cat(grads)  # Store row in Jacobian

    return J

def compute_loss_jacobian_params_classifier(model, x_train, y_train, criterion=None):
    """
    Computes the Jacobian of the per-datum loss w.r.t. model parameters
    for a classification model (e.g. FC_2D_Net on MNIST or 2D toy data).

    Args:
        model: PyTorch classification model.
        x_train: Input data (torch.Tensor), shape (N, D).
        y_train: Ground-truth labels (torch.Tensor), shape (N,).
        criterion: Loss function (e.g., nn.CrossEntropyLoss with reduction='sum').

    Returns:
        J_L: Jacobian matrix (N, P) where:
             - N = number of samples
             - P = number of model parameters
    """
    model.eval()
    device = next(model.parameters()).device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    num_samples = x_train.shape[0]
    num_params = sum(p.numel() for p in model.parameters())
    J_L_theta = torch.zeros(num_samples, num_params, device=device)

    for i in range(num_samples):
        model.zero_grad()
        xi = x_train[i].unsqueeze(0)      # Shape: (1, D)
        yi = y_train[i].unsqueeze(0)      # Shape: (1,)
        
        output = model(xi)                # Shape: (1, num_classes)
        loss = criterion(output, yi)      # Scalar loss
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

        grads = [g.view(-1) for g in grads]
        J_L_theta[i] = torch.cat(grads)

    return J_L_theta

def sample_from_posterior(theta_map, covariance, num_samples=10, scale=0.1):
    """
    Samples parameters from the posterior using SVD for stability
    with added scaling to control uncertainty spread.
    Args:
        theta_map: Flattened MAP estimate of model parameters.
        covariance: Posterior covariance matrix.
        num_samples: Number of samples to draw.
        scale: Scaling factor for eigenvalues.
    Returns:
        samples: List of sampled parameter tensors.
    """
    U, S, V = torch.svd(covariance)
    print(f"Number of clamped eigenvalues: {(S < 1e-6).sum().item()}")
    S = torch.clamp(S, min=1e-6)  # Avoid near-zero eigenvalues
    L = U @ torch.diag(torch.sqrt(scale * S))  # Scale eigenvalues down

    samples = []

    for _ in range(num_samples):
        epsilon = torch.randn_like(theta_map)  # Sample from N(0, I)
        theta_sample = theta_map + L @ epsilon  # Apply scaled transformation
        samples.append(theta_sample)

    return samples

# This is not used anymore -> use the matrix-free version instead
def project_delta_into_nullspace(delta, M, damping=1e-4):
    try:
        MMt = M @ M.T
        inv = torch.linalg.inv(MMt + damping * torch.eye(MMt.shape[0], device=M.device))
        projection = M.T @ (inv @ (M @ delta))
        return delta - projection
    except Exception as e:
        print(f"⚠️ Projection failed: {e}")
        return delta

# This is not used anymore -> use the matrix-free version instead
def alternating_projections_qproj_classifier(
    model, x_train, alpha=10.0, num_samples=100, num_iters=20, batch_size=32
):
    """
    Samples from the projected posterior using alternating projections.

    Args:
        model: Trained classifier (FC_2D_Net).
        x_train: Training inputs.
        alpha: Prior precision (controls posterior spread).
        num_samples: Number of posterior samples.
        num_iters: Number of alternating projection rounds.
        batch_size: Minibatch size for local projections.

    Returns:
        List of sampled weight vectors (torch.Tensor).
    """
    device = next(model.parameters()).device
    x_train = x_train.to(device)
    model.eval()

    P = sum(p.numel() for p in model.parameters())  # number of parameters
    theta_map = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    train_loader = DataLoader(x_train, batch_size=batch_size, shuffle=True)

    samples = []

    for sample_idx in range(num_samples):
        print(f"\n📦 Sampling posterior sample {sample_idx + 1}/{num_samples}")
        delta = torch.randn(P, device=device) / torch.sqrt(torch.tensor(alpha))

        for iter_idx in range(num_iters):
            print(f"  🔁 Iteration {iter_idx + 1}/{num_iters} | delta norm: {delta.norm().item():.4f}")
            for batch_idx, xb in enumerate(train_loader):
                xb = xb.to(device)
                xb.requires_grad_(True)

                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                N, O = probs.shape

                # Construct batch-level H_sqrt blocks
                Hb_blocks = []
                for i in range(N):
                    p = probs[i].unsqueeze(1)
                    Hi = torch.diag_embed(probs[i]) - p @ p.T
                    eigvals, eigvecs = torch.linalg.eigh(Hi)
                    eigvals_clamped = torch.clamp(eigvals, min=1e-6)
                    sqrt_Hi = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.T
                    Hb_blocks.append(sqrt_Hi)

                Hb_sqrt = torch.block_diag(*Hb_blocks)

                # Compute Jacobian of logits wrt parameters
                try:
                    J_b = compute_model_jacobian_params_classifier(model, xb)  # Shape: (N*O, P)
                except Exception as e:
                    print(f"    ⚠️ Skipping batch {batch_idx} due to Jacobian error: {e}")
                    continue

                Mb = Hb_sqrt @ J_b  # (N·O, P)

                delta = project_delta_into_nullspace(delta, Mb)
                if batch_idx % 5 == 0:
                    print(f"    ✅ Batch {batch_idx}: projection applied, delta norm: {delta.norm().item():.4f}")

        final_sample = theta_map + delta.detach().clone()
        print(f"✅ Finished sample {sample_idx + 1}, final delta norm: {delta.norm().item():.4f}")
        samples.append(final_sample)

    print("\n🎉 All posterior samples generated.")
    return samples

def compute_loss_kernel_vp_vmap(loss_fn, model, x, y, params, W):  # W: (B, B)
    """
    Vectorized version: computes JJᵗ W using vmap over rows of W.
    Each row of W is a vector w_i for computing JJᵗ w_i.
    """
    def loss_vector(params_vec):
        out = functional_call(model, params_vec, (x,))  # (B, C)
        return loss_fn(out, y, reduction='none')  # (B,)

    # It is fixed because it depends on the loss function and model, not on the W input
    _, vjp_fun = vjp(loss_vector, params)
    # Compute v = Jᵗ w_row for each w_row in W (VJP)
    Jt_W = vmap(lambda w_row: vjp_fun(w_row)[0])(W)  # tuple((B, P),)

    # Now J(Jᵗ w_i) via JVP for each v_i = Jᵗ w_i
    def jvp_fn(v_i):
        _, JJt_wi = jvp(loss_vector, (params,), (v_i,))
        return JJt_wi

    JJt_W = vmap(jvp_fn)(Jt_W)  # (B, B)
    return JJt_W

@torch.no_grad()
def precompute_loss_ggn_inverse(model, loss_fn, xb, yb, params, damping=1e-3):
    B = xb.shape[0]
    I = torch.eye(B, device=xb.device)

    JJt = compute_loss_kernel_vp_vmap(loss_fn, model, xb, yb, params, I)
    JJt = JJt.detach() + damping * torch.eye(B, device=xb.device)

    eigvals, eigvecs = torch.linalg.eigh(JJt)
    threshold = 1e-3
    inv_eigvals = 1.0 / eigvals
    inv_eigvals[eigvals < threshold] = 0.0

    return eigvecs, inv_eigvals

def project_delta_matrix_free(
    model,
    xb,
    yb,
    delta,
    eigvecs,
    inv_eigvals,
    loss_fn,
    theta
):
    def batch_loss(params):
        out = functional_call(model, params, (xb,))
        return loss_fn(out, yb, reduction='none')  # (B,)
    
    # JVP: J · delta
    _, Jv_dict = jvp(batch_loss, (theta,), (delta,))  # (B,)
    flat_Jv, _ = tree_flatten(Jv_dict)   # list of tensors, P total

    Jv = torch.cat([t.flatten() for t in flat_Jv])  # Shape: (P,), Tensor

    # Low-rank solve: V Λ⁻¹ Vᵗ Jv
    JJt_inv_Jv = eigvecs.T @ Jv            # (r,)
    JJt_inv_Jv = eigvecs @ (JJt_inv_Jv * inv_eigvals)  # (B,)

    # VJP: Jᵗ · v
    _, vjp_fun = vjp(batch_loss, theta)
    Jt_JJt_inv_Jv_dict = vjp_fun(JJt_inv_Jv)[0]

    # Projected delta: delta - Jᵗ(JJt⁻¹ Jv)
    projected_delta = {
        k: delta[k] - Jt_JJt_inv_Jv_dict[k]
        for k in delta
    }

    return projected_delta

def build_projection_operator(model, loss_fn, theta_map, precomputed_eigens):
    def project_fn(delta):
        for xb, yb, eigvecs, inv_eigvals in precomputed_eigens:
            delta = project_delta_matrix_free(
                model=model,
                xb=xb,
                yb=yb,
                delta=delta,
                eigvecs=eigvecs,
                inv_eigvals=inv_eigvals,
                loss_fn=loss_fn,
                theta=theta_map
            )
        return delta
    return project_fn

@torch.no_grad()
def estimate_trace_hutchinson(projection_fn, P, K, param_shapes, device):
    trace_estimates = []
    numels = [torch.tensor(shape).prod().item() for shape in param_shapes.values()]

    for _ in range(K):
        v = torch.randn(P, device=device)  # standard normal vector
        v_split = list(torch.split(v, numels))

        # Reshape each piece to match the original shape
        v_struct_dict = {
            name: part.view(shape)
            for part, (name, shape) in zip(v_split, param_shapes.items())
        }

        proj_v_structured = projection_fn(v_struct_dict)
        proj_v_flat, _ = tree_flatten(proj_v_structured)
        proj_v = torch.cat([x.flatten() for x in proj_v_flat]).to(device)

        trace_estimates.append(v.dot(proj_v).item())  # v^T (UU^T) v
    return sum(trace_estimates) / K

def alternating_projections_qloss_classifier(
    model, dataset, alpha=None, num_samples=100, max_iters=100, rel_tol=1e-4, batch_size=64
):
    """
    Samples from the projected posterior using alternating projections (q_loss),
    using a dataset object with (x, y) pairs.
    
    Args:
        model: Trained classifier (e.g., FC_2D_Net).
        dataset: Dataset object containing (x, y) pairs.
        alpha: Prior precision (controls posterior spread).
        num_samples: Number of posterior samples.
        max_iters: Maximum number of iterations for convergence.
        rel_tol: Relative tolerance for convergence.
        batch_size: Minibatch size for local projections.
    
    Returns:
        List of sampled weight vectors (posterior samples).
    """
    device = next(model.parameters()).device
    model.eval()

    P = sum(p.numel() for p in model.parameters())
    theta_map = dict(model.named_parameters())
    param_shapes = {name: param.shape for name, param in theta_map.items()}
    flat_theta, theta_spec = tree_flatten(theta_map)
    theta_tensor = torch.cat([t.flatten() for t in flat_theta])
    numels = [p.numel() for p in flat_theta]

    delta = tree_unflatten(
        list(torch.randn(P, device=device).split(numels)), theta_spec
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    samples = []
    loss_fn = torch.nn.functional.cross_entropy
    
    # Precompute GGN eigenvectors (matrix-free)
    precompute_ggn_eigvecs_time_start = time.time()
    print("🔄 Precomputing GGN eigenvectors...")
    precomputed_eigens = []
    #num_batches = len(dataset) // batch_size
    #counter = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        eigvecs, inv_eigvals = precompute_loss_ggn_inverse(
            model=model, loss_fn=loss_fn, xb=xb, yb=yb, params=theta_map
        )
        precomputed_eigens.append((xb, yb, eigvecs, inv_eigvals))
        #counter += 1
        #print(f"Precomputed GGN for {counter}/{num_batches} batches")

    print("✅ Precomputation complete.")
    print(f"Time taken for precomputation: {time.time() - precompute_ggn_eigvecs_time_start:.2f} seconds")

    projection_fn = build_projection_operator(
        model=model,
        loss_fn=loss_fn,
        theta_map=theta_map,
        precomputed_eigens=precomputed_eigens
    )

    if alpha is None:
        alpha_estimation_time_start = time.time()
        print("Calculating Alpha with Hutchinson trace estimation...")
        trace_est = estimate_trace_hutchinson(projection_fn, P, K=10, param_shapes=param_shapes, device=device)
        alpha = theta_tensor.norm().item() ** 2 / (P - trace_est)
        print(f"📐 Hutchinson trace: {trace_est:.2f} → optimal alpha: {alpha:.4f}")
        print(f"Time taken for alpha estimation: {time.time() - alpha_estimation_time_start:.2f} seconds")
    
    for sample_idx in range(num_samples):
        print(f"\n📦 Sampling (q_loss) {sample_idx + 1}/{num_samples}")
        delta_init = torch.randn(P, device=device) / torch.sqrt(torch.tensor(alpha))
        delta_split = list(delta_init.split(numels))  # 1D splits
        delta = {
            k: t.view(theta_map[k].shape)  # reshape each split to original shape
            for t, k in zip(delta_split, theta_map.keys())
        }

        for iter_idx in range(max_iters):
            delta_old = delta
            delta = projection_fn(delta)

            flat_delta_old, _ = tree_flatten(delta_old)
            flat_delta_new, _ = tree_flatten(delta)
            delta_new_tensor = torch.cat([t.flatten() for t in flat_delta_new])
            delta_old_tensor = torch.cat([t.flatten() for t in flat_delta_old])

            diff = (delta_new_tensor - delta_old_tensor).norm()
            print(f"  🔁 Iter {iter_idx + 1} | Δdelta norm: {diff:.6f} | delta norm: {delta_new_tensor.norm():.4f}")

            if diff < rel_tol:
                print(f"  ✅ Converged at iteration {iter_idx + 1}")
                break
        
        theta_sample_dict = {k: theta_map[k] + delta[k] for k in delta}

        # Flatten it back to match theta_map
        flat_sample, _ = tree_flatten(theta_sample_dict)
        samples.append(torch.cat([x.flatten() for x in flat_sample]))
        print(f"✅ Done (q_loss) {sample_idx + 1}, norm: {delta_new_tensor.norm().item():.4f}")

    print("\n🎉 All q_loss samples complete.")
    return samples

def save_metrics_regression(y_test_pred, mean_pred, var_pred, y_test_true, path):
    # MAP metrics
    rmse_map = compute_rmse(y_test_true, y_test_pred)

    # Bayesian model metrics
    rmse_bayesian = compute_rmse(y_test_true, mean_pred)
    nll_bayesian = compute_regression_nll(mean_pred, np.sqrt(var_pred), y_test_true)
    ece_bayesian, coverage_per_alpha = compute_regression_ece(mean_pred, np.sqrt(var_pred), y_test_true)
    sharpness_bayesian = compute_sharpness(np.sqrt(var_pred))
    coverage__bayesian = compute_coverage(mean_pred, np.sqrt(var_pred), y_test_true)

    print(f"RMSE (MAP Model): {rmse_map:.4f}")
    print(f"RMSE (Bayesian Mean Model): {rmse_bayesian:.4f}")
    print(f"NLL (Bayesian Model): {nll_bayesian:.4f}")
    print(f"ECE (Bayesian Model): {ece_bayesian:.4f}")

    # Save the metrics
    metrics = {
        "MAP": {
            "RMSE": float(rmse_map)
        },
        "Bayesian": {
            "RMSE": float(rmse_bayesian),
            "NLL": float(nll_bayesian),
            "ECE": float(ece_bayesian),
            "Sharpness": float(sharpness_bayesian),
            "Coverage@90": float(coverage__bayesian),
            "Coverage_per_alpha": [
                {
                    "confidence": float(alpha),
                    "empirical_coverage": float(cov)
                }
                for alpha, cov in coverage_per_alpha
            ]
        }
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

def save_metrics_classification(y_test_pred_map, mean_probs_pred, y_test_true, path):
    # MAP metrics
    acc_map = compute_accuracy(y_test_true, y_test_pred_map)
    nll_map = compute_classification_nll(y_test_true, y_test_pred_map)
    ece_map, mce_map = compute_classification_ece_mce(y_test_true, y_test_pred_map, n_bins=10)
    brier_score_map = compute_brier_score(y_test_true, y_test_pred_map)
    confidence_map = compute_confidence(y_test_pred_map)
    predictive_entropy_map = compute_predictive_entropy(y_test_pred_map)
    auroc_per_class_map = compute_multiclass_auroc(y_test_true, y_test_pred_map)

    # Bayesian model metrics
    acc_bayesian = compute_accuracy(y_test_true, mean_probs_pred)
    nll_bayesian = compute_classification_nll(y_test_true, mean_probs_pred)
    ece_bayesian, mce_bayesian = compute_classification_ece_mce(y_test_true, mean_probs_pred, n_bins=10)
    brier_score_bayesian = compute_brier_score(y_test_true, mean_probs_pred)
    confidence_bayesian = compute_confidence(mean_probs_pred)
    predictive_entropy_bayesian = compute_predictive_entropy(mean_probs_pred)
    aurco_per_class_bayesian = compute_multiclass_auroc(y_test_true, mean_probs_pred)

    print(f"Accuracy (MAP Model): {acc_map:.4f}")
    print(f"Accuracy (Bayesian Mean Model): {acc_bayesian:.4f}")
    print(f"NLL (Bayesian Model): {nll_bayesian:.4f}")
    print(f"NLL (MAP Model): {nll_map:.4f}")
    print(f"ECE (Bayesian Model): {ece_bayesian:.4f}")
    print(f"ECE (MAP Model): {ece_map:.4f}")

    # Save the metrics
    metrics = {
        "MAP": {
            "Accuracy": float(acc_map),
            "NLL": float(nll_map),
            "ECE": float(ece_map),
            "MCE": float(mce_map),
            "Brier Score": float(brier_score_map),
            "Confidence": float(confidence_map),
            "Predictive Entropy": float(predictive_entropy_map),
            "AUROC per class": [
                {
                    "class": str(key),
                    "auroc": float(value)
                }
                for key, value in auroc_per_class_map.items()
            ]
        },
        "Bayesian": {
            "Accuracy": float(acc_bayesian),
            "NLL": float(nll_bayesian),
            "ECE": float(ece_bayesian),
            "MCE": float(mce_bayesian),
            "Brier Score": float(brier_score_bayesian),
            "Confidence": float(confidence_bayesian),
            "Predictive Entropy": float(predictive_entropy_bayesian),
            "AUROC per class": [
                {
                    "class": str(key),
                    "auroc": float(value)
                }
                for key, value in aurco_per_class_bayesian.items()
            ]
        }
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics