import ctypes
import os
from pathlib import Path

SYS_CREATE_RULESET, SYS_ADD_RULE, SYS_RESTRICT_SELF = 444, 445, 446
RULE_PATH_BENEATH, RULE_NET_PORT = 1, 2
PR_SET_NO_NEW_PRIVS = 38

EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
DIR_ONLY = sum(1 << i for i in range(4, 13))
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
FILE_RIGHTS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV
BIND_TCP, CONNECT_TCP = 1 << 0, 1 << 1


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def _check(rc: int, what: str) -> int:
    if rc < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"landlock {what}: {os.strerror(err)}")
    return rc


def abi() -> int:
    return _libc.syscall(SYS_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))


def restrict(read_only: list[str], read_write: list[str]) -> None:
    version = abi()
    if version < 4:
        raise RuntimeError(f"landlock ABI {version} < 4: cannot block files and network, refusing to run agent code")
    handled = EXECUTE | WRITE_FILE | READ_FILE | READ_DIR | DIR_ONLY | REFER | TRUNCATE
    if version >= 5:
        handled |= IOCTL_DEV
    attr = RulesetAttr(handled, BIND_TCP | CONNECT_TCP)
    fd = _check(_libc.syscall(SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0)), "create_ruleset")
    ro = EXECUTE | READ_FILE | READ_DIR
    for paths, rights in ((read_only, ro), (read_write, handled)):
        for p in paths:
            try:
                path_fd = os.open(p, os.O_PATH | os.O_CLOEXEC)
            except FileNotFoundError:
                continue
            allowed = rights & handled
            if not Path(p).is_dir():
                allowed &= FILE_RIGHTS
            rule = PathBeneathAttr(allowed, path_fd)
            _check(_libc.syscall(SYS_ADD_RULE, ctypes.c_int(fd), ctypes.c_int(RULE_PATH_BENEATH), ctypes.byref(rule), ctypes.c_uint32(0)), f"add_rule {p}")
            os.close(path_fd)
    _check(_libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
    _check(_libc.syscall(SYS_RESTRICT_SELF, ctypes.c_int(fd), ctypes.c_uint32(0)), "restrict_self")
    os.close(fd)



if __name__ == "__main__":
    print(f"landlock ABI {abi()} (the harness needs 4 or newer)")
