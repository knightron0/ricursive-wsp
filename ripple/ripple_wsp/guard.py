def _banned(*args, **kwargs):
    raise RuntimeError("gradients, autograd and optimizers are not allowed in this task")


class _BannedOptimizer:
    def __init__(self, *args, **kwargs):
        _banned()


def lock_torch(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    no_grad = torch.no_grad

    def set_grad_enabled(mode=False):
        if mode:
            _banned()
        return no_grad()

    torch.set_grad_enabled = set_grad_enabled
    torch.enable_grad = _banned
    torch.Tensor.backward = _banned
    torch.Tensor.register_hook = _banned
    torch.autograd.backward = _banned
    torch.autograd.grad = _banned
    torch.autograd.functional.jacobian = _banned
    torch.autograd.functional.hessian = _banned
    torch.autograd.functional.vjp = _banned
    torch.autograd.functional.jvp = _banned
    for name in ("grad", "grad_and_value", "vjp", "jvp", "jacrev", "jacfwd", "hessian", "linearize"):
        if hasattr(torch.func, name):
            setattr(torch.func, name, _banned)
    for name in dir(torch.optim):
        obj = getattr(torch.optim, name)
        if isinstance(obj, type) and issubclass(obj, torch.optim.Optimizer):
            setattr(torch.optim, name, _BannedOptimizer)
    lock_solvers(torch)


def _no_fit(*args, **kwargs):
    raise RuntimeError("linear solvers are disabled: weights may not be fitted to data in this task")


def lock_solvers(torch) -> None:
    import numpy as np

    for name in ("solve", "lstsq", "pinv", "inv", "tensorsolve", "tensorinv"):
        setattr(np.linalg, name, _no_fit)
    for name in ("solve", "solve_ex", "lstsq", "pinv", "inv", "inv_ex", "tensorsolve", "tensorinv", "lu_solve", "solve_triangular"):
        if hasattr(torch.linalg, name):
            setattr(torch.linalg, name, _no_fit)
    for name in ("cholesky_solve", "triangular_solve", "lu_solve", "inverse", "pinverse", "lstsq"):
        if hasattr(torch, name):
            setattr(torch, name, _no_fit)
    torch.Tensor.inverse = _no_fit
    torch.Tensor.pinverse = _no_fit
