from torch.func import jvp, vjp
from torch.autograd.functional import jacobian
import torch

def loss_kernel_vp( # loss_kernel_vector_product = JJt @ W
    loss_lmbd,
    w,
    params
):
    _, Jtw_fn = vjp(loss_lmbd, params)
    JtW = Jtw_fn(w.reshape((-1,)))[0]
    _, JJtW = jvp(loss_lmbd, (params,), (JtW,))
    return JJtW

def precompute_loss_inv(
    model_fn,
    loss_fn, 
    params_vec, 
    train_data
):
    device = params_vec.device

    def compute_eigendecomp_for_batch(x, y):
        loss_lmbd = lambda p_vec: loss_fn(model_fn(p_vec, x), y)
        kernel_vp = lambda w: loss_kernel_vp(loss_lmbd, w, params_vec)
        batch_size = x.shape[0]
        w = torch.ones((batch_size,)).to(device)  # Vector for vector product
        JJt = jacobian(kernel_vp, w, create_graph=False)
        JJt = JJt.reshape(batch_size, batch_size)
        eigvals, eigvecs = torch.linalg.eigh(JJt)
        idx = eigvals < 1e-3
        inv_eigvals = torch.where(idx, 1., eigvals)
        inv_eigvals = 1/inv_eigvals
        inv_eigvals = torch.where(idx, 0., inv_eigvals)
        del loss_lmbd, kernel_vp, JJt

        torch.cuda.empty_cache()
        return eigvecs, inv_eigvals
    
    eigvecs_list = []
    inv_eigvals_list = []

    for i, (x, y) in enumerate(train_data):
        x, y = x.to(device), y.to(device)
        eigvecs_i, inv_eigvals_i = compute_eigendecomp_for_batch(x, y)
        eigvecs_list.append(eigvecs_i)
        inv_eigvals_list.append(inv_eigvals_i)
        torch.cuda.empty_cache()

    return eigvecs_list, inv_eigvals_list

