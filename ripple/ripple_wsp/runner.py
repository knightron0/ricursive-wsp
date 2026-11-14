import os
import runpy
import sys
from pathlib import Path

from ripple_wsp.guard import lock_torch
from ripple_wsp.sandbox import restrict

ROOT = Path(__file__).resolve().parents[1]


def lock(writable: str, readable: list[str]) -> None:
    if os.environ.get("RIPPLE_WSP_NO_LANDLOCK") == "1":
        print("WARNING: RIPPLE_WSP_NO_LANDLOCK=1, agent code runs without the file/network sandbox", file=sys.stderr, flush=True)
        return
    read_only = [
        str(Path(sys.prefix).resolve()),
        str(Path(sys.base_prefix).resolve()),
        str(ROOT),
        "/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc", "/sys",
        f"/proc/{os.getpid()}", "/proc/cpuinfo", "/proc/meminfo", "/proc/stat",
        "/dev/urandom", "/dev/random", "/dev/zero",
        *readable,
    ]
    restrict(read_only, [str(Path(writable).resolve()), "/dev/null", "/dev/shm"])


def main() -> None:
    lock_torch(int(os.environ["LAB_THREADS"]))
    lock(os.getcwd(), [os.environ["LAB_SPLIT"]] if os.environ.get("LAB_SPLIT") else [])
    script = sys.argv[1]
    sys.argv = [script]
    sys.path.insert(0, os.getcwd())
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
