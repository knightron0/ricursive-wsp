"""Run one tools() call: python -m wsp.sandbox WORKSPACE  (code in WORKSPACE/_call.py).

The workspace persists across calls. The harness writes into it only what the mode allows:
nothing (blind), data.pt = {"images", "classes"} (reasoned: labels never exist in this
process), or data.pt = {"images", "labels", "classes"} (full). Plus start.pt, the start weights.

The code runs with `torch`, `math`, `data` (or None) and `start` preloaded: no helper library.
Restrictions (Python audit hooks: process isolation, not a security sandbox):
- No network, subprocesses, threads or ctypes.
- Read only the Python install and the workspace (not the wsp package: no harness code to import);
  write only the workspace.
- No gradients in any mode: grad mode is off for good; autograd, torch.optim, torch.func transforms removed.
"""
import math
import os
from pathlib import Path
import sys

import torch

PACKAGE_DIR = os.path.realpath(Path(__file__).parent)
BLOCKED_EVENTS = {
    'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr',
    'socket.sendto', 'urllib.Request', 'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn',
    'os.spawn', 'os.fork', 'os.forkpty', 'pty.spawn', '_thread.start_new_thread',
    '_thread.start_joinable_thread', 'ctypes.dlopen', 'ctypes.dlsym', 'ctypes.cdata',
}


def block_gradients():
    """Every mode: grad mode off and can't be re-enabled; autograd, torch.optim, torch.func gone."""
    import types
    torch.set_grad_enabled(False)

    def refuse(*args, **kwargs):
        raise RuntimeError('Gradients are disabled in this harness')

    class set_grad_enabled(torch.set_grad_enabled):
        def __init__(self, mode):
            if mode:
                refuse()
            super().__init__(mode)

    class enable_grad:
        def __init__(self, *args, **kwargs):
            refuse()

    torch.set_grad_enabled = torch.autograd.set_grad_enabled = set_grad_enabled
    torch.enable_grad = torch.autograd.enable_grad = torch.autograd.grad_mode.enable_grad = enable_grad
    torch.inference_mode = lambda mode=True: refuse() if not mode else torch.no_grad()
    torch.autograd.grad = torch.autograd.backward = torch.Tensor.backward = refuse
    torch.Tensor.requires_grad_ = lambda self, mode=True: refuse() if mode else self
    for name in ('grad', 'grad_and_value', 'vjp', 'jvp', 'jacrev', 'jacfwd', 'hessian', 'functional_call'):
        if hasattr(torch.func, name):
            setattr(torch.func, name, refuse)
    stub = types.ModuleType('torch.optim')
    stub.__getattr__ = lambda name: refuse()
    torch.optim = sys.modules['torch.optim'] = stub
    sys.modules['torch.optim.lr_scheduler'] = stub


def inside(path, roots):
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def restrict_io(workspace):
    work = os.path.realpath(workspace)
    repo = os.path.dirname(PACKAGE_DIR)  # holds data/ and runs/: never readable
    readable = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path) if p and os.path.isdir(p)}
    readable = {r for r in readable if not inside(repo, {r}) and not inside(work, {r})} | {work, '/dev'}

    def hook(event, args):
        if event in BLOCKED_EVENTS:
            raise PermissionError(f'{event} is not allowed in the sandbox')
        if event == 'open':
            path, mode = args[0], args[1] or 'r'
            if isinstance(path, int) or path is None:
                return
            real = os.path.realpath(os.fsdecode(path))
            if any(c in str(mode) for c in 'wax+'):
                if not inside(real, {work, '/dev'}):
                    raise PermissionError(f'Writing {path} is not allowed')
            elif not inside(real, readable):
                raise PermissionError(f'Reading {path} is not allowed')

    sys.addaudithook(hook)


def main(workspace):
    work = Path(workspace).resolve()
    os.chdir(work)
    start = torch.load(work / 'start.pt', weights_only=True)
    data = torch.load(work / 'data.pt', weights_only=True) if (work / 'data.pt').exists() else None
    source = (work / '_call.py').read_text()
    (work / '_call.py').unlink()
    block_gradients()
    restrict_io(str(work))
    namespace = {'__name__': '__main__', 'torch': torch, 'math': math, 'data': data, 'start': start}
    exec(compile(source, 'tools_call.py', 'exec'), namespace)


if __name__ == '__main__':
    main(*sys.argv[1:])
