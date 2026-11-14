"""Subprocess entry point: run one weight-building program under restrictions.

Usage: python -m wsp.apply WORKDIR DATA.pt [--allow-gradients]
WORKDIR holds input.pt (weights), program.py and info.json; output.pt is written there.

Restrictions (best effort; process isolation, not a security sandbox):
- No network, subprocesses or new threads (audit hook).
- Files: read only Python/torch installs, this package and WORKDIR/DATA; write only WORKDIR.
  So the program sees only the X% data subset it was given, never the full dataset.
- No gradients: grad mode is off and cannot be re-enabled; torch.optim and
  torch.func gradient transforms are removed.
"""
import json
import math
import os
from pathlib import Path
import sys
import types

import torch

PACKAGE_DIR = os.path.realpath(Path(__file__).parent)
BLOCKED_EVENTS = {
    'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.gethostbyname',
    'socket.gethostbyaddr', 'socket.sendto', 'urllib.Request',
    'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn', 'os.fork',
    'os.forkpty', 'pty.spawn', '_thread.start_new_thread', '_thread.start_joinable_thread',
}


def block_gradients():
    torch.set_grad_enabled(False)

    def refuse(*args, **kwargs):
        raise RuntimeError('Gradients are disabled: compute weights analytically from data')

    original = torch.set_grad_enabled

    class set_grad_enabled(original):
        def __init__(self, mode):
            if mode:
                refuse()
            super().__init__(mode)

    class enable_grad:
        def __init__(self, *args, **kwargs):
            refuse()

    torch.set_grad_enabled = torch.autograd.set_grad_enabled = set_grad_enabled
    torch.enable_grad = torch.autograd.enable_grad = torch.autograd.grad_mode.enable_grad = enable_grad
    torch.autograd.grad = torch.autograd.backward = torch.Tensor.backward = refuse
    for name in ('grad', 'grad_and_value', 'vjp', 'jvp', 'jacrev', 'jacfwd', 'hessian', 'functional_call'):
        if hasattr(torch.func, name):
            setattr(torch.func, name, refuse)
    stub = types.ModuleType('torch.optim')
    stub.__getattr__ = lambda name: refuse()
    torch.optim = sys.modules['torch.optim'] = stub


def restrict_io(workdir, data_path):
    readable = {os.path.realpath(p) for p in
                {sys.prefix, sys.base_prefix, sys.exec_prefix, PACKAGE_DIR, workdir, '/dev'}}
    readable |= {os.path.realpath(p) for p in sys.path if p and os.path.isdir(p)
                 and not os.path.realpath(p) == os.path.realpath(os.path.dirname(PACKAGE_DIR))}
    data_real = os.path.realpath(data_path)
    work_real = os.path.realpath(workdir)

    def inside(path, roots):
        return any(path == r or path.startswith(r + os.sep) for r in roots)

    def hook(event, args):
        if event in BLOCKED_EVENTS:
            raise PermissionError(f'{event} is not allowed in weight programs')
        if event == 'open':
            path, mode = args[0], args[1] or 'r'
            if isinstance(path, int) or path is None:
                return
            real = os.path.realpath(os.fsdecode(path))
            writing = any(c in str(mode) for c in 'wax+')
            if writing and not inside(real, {work_real, '/dev'}):
                raise PermissionError(f'Writing {path} is not allowed')
            if not writing and real != data_real and not inside(real, readable):
                raise PermissionError(f'Reading {path} is not allowed; use the `data` argument')

    sys.addaudithook(hook)


def main(workdir, data_path, *flags):
    workdir = Path(workdir)
    weights = torch.load(workdir / 'input.pt', weights_only=True)
    data = torch.load(data_path, weights_only=True)
    info = json.loads((workdir / 'info.json').read_text())
    source = (workdir / 'program.py').read_text()
    from wsp import toolkit  # import before restrictions so torch internals load normally
    sys.modules['toolkit'] = toolkit  # programs may write `import toolkit`
    toolkit.capture(weights, torch.zeros(2, 3, 32, 32, dtype=torch.uint8), ['fc'],
                    info['num_classes'])  # warm up device (works even with no data)
    if '--allow-gradients' not in flags:
        block_gradients()
    restrict_io(str(workdir), data_path)
    namespace = {'__name__': 'weight_program', 'torch': torch, 'math': math, 'toolkit': toolkit}
    torch.manual_seed(0)
    exec(compile(source, 'program.py', 'exec'), namespace)
    build = namespace.get('build')
    if not callable(build):
        raise SystemExit('Program must define build(weights, data, info)')
    result = build(weights, data, info)
    if not isinstance(result, dict):
        raise SystemExit(f'build() must return a dict of tensors, got {type(result).__name__}')
    torch.save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()},
               workdir / 'output.pt')


if __name__ == '__main__':
    main(*sys.argv[1:])
