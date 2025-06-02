import torch
from torch.func import jvp, vjp

def loss_projection(
    delta,
    model_fn, 
    loss_fn,
    params_vec,
    projection_data,
    batched_eigvecs,
    batched_inv_eigvals,
    num_iters,
    x_val
):
    device = params_vec.device
    
    def loss_projection_step(delta, x, y, eigvecs, inv_eigvals):
        loss_lmbd = lambda p_vec: loss_fn(model_fn(p_vec, x), y)

        with torch.no_grad():
            _, Jv = jvp(loss_lmbd, (params_vec,), (delta,))

            JJt_inv_Jv = eigvecs.T @ Jv
            JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
            JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0],))

            _, Jtv_fn = vjp(loss_lmbd, params_vec)
            Jt_JJt_inv_Jv = Jtv_fn(JJt_inv_Jv)[0]

            proj_delta = delta - Jt_JJt_inv_Jv

        return proj_delta
    
    def project_through_data(iter, iter_delta):
        for i, (x, y) in enumerate(projection_data):
            x, y = x.to(device), y.to(device)
            eigvecs = batched_eigvecs[i]
            inv_eigvals = batched_inv_eigvals[i]
            iter_delta = loss_projection_step(iter_delta, x, y, eigvecs, inv_eigvals)

        # Compute norm of the projection
        proj_vector = iter_delta
        proj_norm = torch.linalg.vector_norm(proj_vector)

        # Compute kernel norm at validation point
        model_out_fn = lambda p: model_fn(p, x_val)
        _, Jdelta = jvp(model_out_fn, (params_vec,), (iter_delta,))
        kernel_norm = torch.linalg.vector_norm(Jdelta)

        # FOR SERIALIZED VERSION DEBUGGING
        #print(f"Iteration {iter}: Proj Norm = {proj_norm.item():.4e} | Kernel Norm Ratio = {(kernel_norm.item()/proj_norm.item()):.4e}")
        
        return iter_delta, proj_norm, kernel_norm
    
    proj_norms = []
    kernel_norm_ratios = []

    for i in range(num_iters):
        delta, proj_norm, kernel_norm = project_through_data(i, delta)
        proj_norms.append(proj_norm)
        kernel_norm_ratios.append(kernel_norm / proj_norm)

    proj_norms = torch.stack(proj_norms)
    kernel_norm_ratios = torch.stack(kernel_norm_ratios)

    return (
        delta,
        proj_norms,
        kernel_norm_ratios
    )
    
