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


def sample_loss_projections_dataloader(
    model,
    train_dataset, 
    num_samples=5,
    max_iters=1000, 
    macro_batch_size=32,
    sample_batch_size=16
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
    projection_data = DataLoader(train_dataset, batch_size=macro_batch_size, shuffle=False, drop_last=True)

    # Initialize prior samples from a univariate Gaussian
    projected_samples = torch.randn(num_samples, n_params, device=params_vec.device)
    eps = projected_samples
    x_val, y_val = next(iter(DataLoader(train_dataset, batch_size=macro_batch_size)))
    x_val, y_val = x_val.to(device), y_val.to(device)

    for i, batch in enumerate(projection_data):
        x_data, y_data = batch
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        y_train_batched = y_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + y_data.shape[1:])
        batched_eigvecs, batched_inv_eigvals = precompute_loss_inv(
            model_fn, 
            loss_fn, 
            params_vec, 
            zip(x_train_batched, y_train_batched)
        )
        
        loss_projection_fn = lambda init_delta: loss_projection(
            delta=init_delta,
            model_fn=model_fn,
            loss_fn=loss_fn,
            params_vec=params_vec,
            projection_data=zip(x_train_batched, y_train_batched),
            batched_eigvecs=batched_eigvecs,
            batched_inv_eigvals=batched_inv_eigvals,
            num_iters=max_iters,
            x_val=x_val
        )
        projected_samples, proj_norms, kernel_norm_ratios  = vmap(loss_projection_fn)(projected_samples)
        if i % 100 == 0:
            print(f"Iteration {i+1}/{len(projection_data)}: Kernel Ratio = {kernel_norm_ratios.mean().item():.4e} | Proj Norm = {proj_norms.mean().item():.4e}")
        del x_train_batched, x_data, y_train_batched, y_data, batched_eigvecs, batched_inv_eigvals, loss_projection_fn
    
    final_proj_norms = proj_norms[:, -1]               # Shape: (num_samples,)
    final_kernel_ratios = kernel_norm_ratios[:, -1]    # Shape: (num_samples,)

    for i in range(final_proj_norms.shape[0]):
        print(f"Sample {i}: Final Proj Norm = {final_proj_norms[i].item():.4e} | "
            f"Final Kernel Ratio = {final_kernel_ratios[i].item():.4e}")
    
    print(f"Calculating optimal alpha...")
    trace_proj = (vmap(lambda e, x: torch.dot(e, x), in_dims=(0,0))(eps, projected_samples)).mean()
    alpha = torch.dot(params_vec, params_vec) / (n_params - trace_proj)
    print(f"Trace of projection: {trace_proj.item():.4e}")
    print("Optimal alpha:", alpha) 

    posterior_samples = vmap(lambda single_sample: params_vec + (1./torch.sqrt(alpha)) * single_sample)(projected_samples)

    return posterior_samples