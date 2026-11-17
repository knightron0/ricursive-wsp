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


# Audit events that touch the filesystem by path: event -> (arg indices read, arg indices changed, dir_fd index).
# Changes are confined to the workspace; listings and reads to the readable roots.
PATH_EVENTS = {
    'os.listdir': ((0,), (), None), 'os.scandir': ((0,), (), None), 'os.chdir': ((), (0,), None),
    'os.listxattr': ((0,), (), None),
    'os.remove': ((), (0,), 1), 'os.rmdir': ((), (0,), 1), 'os.mkdir': ((), (0,), 2),
    'os.rename': ((), (0, 1), 2), 'os.link': ((0,), (1,), 2), 'os.symlink': ((0,), (1,), 2),
    'os.truncate': ((), (0,), None), 'os.chmod': ((), (0,), 2), 'os.chown': ((), (0,), 3),
    'os.utime': ((), (0,), 3), 'os.chflags': ((), (0,), None), 'os.lchflags': ((), (0,), None),
    'os.setxattr': ((), (0,), None), 'os.removexattr': ((), (0,), None),
    'shutil.copyfile': ((0,), (1,), None), 'shutil.copytree': ((0,), (1,), None),
    'shutil.move': ((), (0, 1), None), 'shutil.rmtree': ((), (0,), 1), 'shutil.chown': ((), (0,), None),
}


_REAL_STAT = {'stat': os.stat, 'lstat': os.lstat}
_GUARD = {'busy': False}


def realpath(path):
    """os.path.realpath with the unguarded stat functions (it calls os.lstat itself)."""
    if _GUARD['busy']:
        return os.path.realpath(path)
    _GUARD['busy'] = True
    try:
        return os.path.realpath(path)
    finally:
        _GUARD['busy'] = False


def no_dir_fd(fd):
    return fd is None or (isinstance(fd, int) and fd < 0)  # CPython passes a negative AT_FDCWD when unset


def guard_unaudited(readable, work):
    """Calls with no audit event. stat/lstat (and os.path.exists/getsize/...) may only look at readable paths;
    mkfifo/mknod only create inside the workspace. Patched on both os and posix so neither name bypasses it."""
    import posix
    rules = {'stat': (readable, 'Reading'), 'lstat': (readable, 'Reading'),
             'mkfifo': ({work}, 'Creating'), 'mknod': ({work}, 'Creating')}
    for name, (roots, what) in rules.items():
        real_fn = getattr(posix, name, None)
        if real_fn is None:
            continue

        def wrapped(path, *args, _fn=real_fn, _roots=roots, _what=what, **kwargs):
            if not _GUARD['busy'] and not isinstance(path, int):
                if not no_dir_fd(kwargs.get('dir_fd')) or not inside(realpath(os.fsdecode(path)), _roots):
                    raise PermissionError(f'{_what} {path} is not allowed')
            return _fn(path, *args, **kwargs)

        setattr(os, name, wrapped)
        setattr(posix, name, wrapped)


def block_ipc():
    """POSIX shared memory and named semaphores live in a machine-wide namespace: a channel between runs."""
    import importlib

    def refuse(*args, **kwargs):
        raise PermissionError('shared memory and named semaphores are not allowed in the sandbox')

    targets = (('multiprocessing.synchronize', ('SemLock', 'sem_unlink')), ('_multiprocessing', ('SemLock', 'sem_unlink')),
               ('_posixshmem', ('shm_open', 'shm_unlink')))
    mods = {}
    for module, _ in targets:  # import everything first: synchronize reads SemLock attributes at import time
        try:
            mods[module] = importlib.import_module(module)
        except ImportError:
            pass
    for module, names in targets:
        mod = mods.get(module)
        if mod is None:
            continue
        for name in names:
            if hasattr(mod, name):
                setattr(mod, name, refuse)


def inside(path, roots):
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def restrict_io(workspace):
    work = os.path.realpath(workspace)
    repo = os.path.dirname(PACKAGE_DIR)  # holds data/ and runs/: never readable
    readable = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path) if p and os.path.isdir(p)}
    readable = {r for r in readable if not inside(repo, {r}) and not inside(work, {r})} | {work, '/dev'}

    def check(path, roots, what):
        if isinstance(path, int):  # an fd: it came from an open() this hook already checked
            return
        if not inside(realpath(os.fsdecode(path) if path is not None else '.'), roots):
            raise PermissionError(f'{what} {path} is not allowed')

    def hook(event, args):
        if event in BLOCKED_EVENTS:
            raise PermissionError(f'{event} is not allowed in the sandbox')
        if event in PATH_EVENTS:
            reads, writes, fd_at = PATH_EVENTS[event]
            if fd_at is not None and len(args) > fd_at and not no_dir_fd(args[fd_at]):
                raise PermissionError(f'{event} with dir_fd is not allowed in the sandbox')
            for i in reads:
                check(args[i], readable, 'Reading')
            for i in writes:
                check(args[i], {work}, 'Changing')
            return
        if event == 'open':
            path, mode = args[0], args[1] or 'r'
            if isinstance(path, int) or path is None:
                return
            real = realpath(os.fsdecode(path))
            if any(c in str(mode) for c in 'wax+'):
                if not inside(real, {work, '/dev'}):
                    raise PermissionError(f'Writing {path} is not allowed')
            elif not inside(real, readable):
                raise PermissionError(f'Reading {path} is not allowed')

    guard_unaudited(readable, work)
    block_ipc()
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
