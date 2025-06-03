import time
import torch
from torch.func import vmap, functional_call
from torch.utils.data import DataLoader

from alternating_loss_projections import loss_projection
from precompute_loss_inv import precompute_loss_inv
from utils import get_param_vector_tools

def sample_loss_projections(
    model, 
    train_dataset, 
    num_samples=5,
    max_iters=1000, 
    rel_tol=1e-3, 
    batch_size=16
):
    # Prepare function parameters
    loss_fn = lambda pred, target: torch.nn.functional.cross_entropy(pred, target, reduction='none')

    params = dict(model.named_parameters())
    params_vec, unflatten_params = get_param_vector_tools(params)

    # Define stateless model_fn
    def model_fn(params_vec, x):
        return functional_call(model, unflatten_params(params_vec), x)
    
    device = params_vec.device
    n_params = params_vec.shape[0]
    projection_data = list(DataLoader(train_dataset, batch_size=16, shuffle=False))
    
    # Precompute GGN eigenvectors (matrix-free)
    precompute_ggn_eigvecs_time_start = time.time()
    print("🔄 Precomputing GGN eigenvectors...")
    batched_eigvecs, batched_inv_eigvals = precompute_loss_inv(
        model_fn, loss_fn, params_vec, projection_data
    )
    print("✅ Precomputation complete.")
    print(f"Time taken for precomputation: {time.time() - precompute_ggn_eigvecs_time_start:.2f} seconds")

    # Initialize prior samples from a univariate Gaussian
    prior_samples = torch.randn(num_samples, n_params, device=params_vec.device)
    x_val, y_val = next(iter(DataLoader(train_dataset, batch_size=batch_size)))
    x_val, y_val = x_val.to(device), y_val.to(device)
    
    loss_projection_fn = lambda init_delta: loss_projection(
        delta=init_delta,
        model_fn=model_fn,
        loss_fn=loss_fn,
        params_vec=params_vec,
        projection_data=projection_data,
        batched_eigvecs=batched_eigvecs,
        batched_inv_eigvals=batched_inv_eigvals,
        num_iters=max_iters,
        x_val=x_val
    )

    """# SERIALIZED VERSION FOR DEBUGGING
    projected_samples = []
    for i in range(num_samples):
        print(f"Processing sample {i+1}/{num_samples}...")
        start_time = time.time()
        init_delta = torch.randn(params_vec.shape[0], device=device)
        projected_sample, proj_norms, kernel_norm_ratios = loss_projection_fn(init_delta)
        end_time = time.time()
        final_proj_norm = proj_norms[-1]
        final_kernel_ratio = kernel_norm_ratios[-1]
        print(f"Sample {i}: Final Proj Norm = {final_proj_norm.item():.4e} | "
              f"Final Kernel Ratio = {final_kernel_ratio.item():.4e} | "
              f"Time taken: {end_time - start_time:.2f} seconds")
        projected_samples.append(projected_sample)

    posterior_samples = []
    for sample in projected_samples:
        posterior_sample = params_vec + (1. / torch.sqrt(torch.tensor(1.0))) * sample  # Assuming alpha=1 for simplicity
        posterior_samples.append(posterior_sample) """


    projected_samples, proj_norms, kernel_norm_ratios = vmap(loss_projection_fn)(prior_samples)

    final_proj_norms = proj_norms[:, -1]               # Shape: (num_samples,)
    final_kernel_ratios = kernel_norm_ratios[:, -1]    # Shape: (num_samples,)

    for i in range(final_proj_norms.shape[0]):
        print(f"Sample {i}: Final Proj Norm = {final_proj_norms[i].item():.4e} | "
            f"Final Kernel Ratio = {final_kernel_ratios[i].item():.4e}")

    print(f"Calculating optimal alpha...")
    trace_proj = vmap(lambda e, x: torch.dot(e, x), in_dims=(0, 0))(prior_samples, projected_samples).mean()
    alpha = torch.dot(params_vec, params_vec) / (n_params - trace_proj)
    print(f"Trace of projection: {trace_proj.item():.4e}")
    print(f"Optimal alpha: {alpha.item():.4f}")

    posterior_samples = vmap(lambda single_sample: params_vec + (1./torch.sqrt(alpha)) * single_sample)(projected_samples)

    return posterior_samples
