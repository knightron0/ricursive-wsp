import os
import re
import resource

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

FIT_BANNED = [
    (r"\blinalg\s*\.\s*(solve\w*|lstsq|pinv\w*|inv\w*|tensorsolve|tensorinv)\b|\b(cholesky_solve|triangular_solve|lu_solve)\b", "a linear solver (fitting weights to data)"),
]

BANNED = [
    (r"\.backward\s*\(", "a backward pass"),
    (r"\bautograd\b", "torch.autograd"),
    (r"\btorch\s*\.\s*optim\b|\bfrom\s+torch\s+import\s+optim\b|\boptim\s*\.\s*(SGD|Adam\w*|RMSprop|LBFGS|Adagrad|Adadelta|Adamax|ASGD|NAdam|RAdam|Rprop|SparseAdam)\b", "an optimizer"),
    (r"requires_grad\s*=\s*True|requires_grad_\s*\(\s*(True)?\s*\)", "requires_grad"),
    (r"enable_grad|set_grad_enabled|inference_mode\s*\(\s*(mode\s*=\s*)?False", "re-enabling gradients"),
    (r"\btorch\s*\.\s*func\b|\bfrom\s+torch\s+import\s+func\b|\bfunctorch\b", "torch.func transforms"),
    (r"(?m)^\s*(import|from)\s+(jax|tensorflow|keras|sklearn|scipy|torchvision|requests|urllib|http|socket|subprocess)\b", "a disallowed package"),
    (r"\bos\s*\.\s*(system|popen|exec\w*|spawn\w*)\b|\bsubprocess\b", "running other programs"),
    (r"['\"]/(dfs|lfs|home|tmp|proc|sys|etc|usr|var|root|mnt|opt|dev|run|srv|data|scratch)\b", "an absolute path"),
    (r"\.\./", "a parent-directory path"),
    (r"expanduser|Path\s*\.\s*home|\bos\s*\.\s*environ\b|getenv", "the environment or home directory"),
]

FRAME = re.compile(r'^\s*File "([^"]+)", line \d+')


def compress_tracebacks(text: str) -> str:
    out, skipped, in_tb = [], 0, False
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Traceback (most recent call last):"):
            in_tb, skipped = True, 0
            out.append(line)
            i += 1
            continue
        m = FRAME.match(line) if in_tb else None
        if m:
            j = i + 1
            while j < len(lines) and lines[j].startswith("    ") and not FRAME.match(lines[j]):
                j += 1
            path = m.group(1)
            library = "site-packages" in path or "/ripple_wsp/" in path or path.startswith("<frozen")
            if library:
                skipped += 1
            else:
                if skipped:
                    out.append(f"  [{skipped} library frame{'s' if skipped > 1 else ''} omitted]")
                    skipped = 0
                out.extend(lines[i:j])
            i = j
            continue
        if in_tb and not line.startswith(" "):
            if skipped:
                out.append(f"  [{skipped} library frame{'s' if skipped > 1 else ''} omitted]")
            in_tb, skipped = False, 0
        out.append(line)
        i += 1
    return "\n".join(out)


def head_tail(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    head = n // 3
    return text[:head] + f"\n[... {len(text) - n} characters omitted ...]\n" + text[-(n - head):]


def limit_memory():
    os.setsid()
    cap = 64 * 2**30
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
