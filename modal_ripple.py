"""Serve ricursiveAI/Ripple-65B-Preview on Modal (2x B200, SGLang + DFlash2 speculative decoding).

Uses the model repo's own launcher (inference/server/serve.py), which wraps SGLang 0.5.19 and
exposes an OpenAI-compatible API at /v1/chat/completions (served model name: "ripple").

One-time setup:
    pip install modal && modal setup
    modal secret create ripple-api-key RIPPLE_API_KEY=$(openssl rand -hex 24)
    modal run modal_ripple.py::download      # ~130 GB target + draft into a Modal Volume

Serve:
    modal deploy modal_ripple.py             # prints https://<workspace>--ripple-serve.modal.run

Then point agent.py at it (see README):
    export MODAL_RIPPLE_URL=https://<workspace>--ripple-serve.modal.run/v1
    export MODAL_RIPPLE_API_KEY=<same key as the secret>
    python agent.py --provider modal --task scratch-cifar10 --data-fraction 0.1 --rounds 20
"""
import os
import subprocess

import modal

TARGET = 'ricursiveAI/Ripple-65B-Preview'
DRAFT = 'ricursiveAI/Ripple-65B-Preview-DFlash2'
MODELS = '/models'
TARGET_DIR = f'{MODELS}/Ripple-65B-Preview'
DRAFT_DIR = f'{MODELS}/Ripple-65B-Preview-DFlash2'
PORT = 30000

# GPU profile, chosen at deploy time: RIPPLE_GPU=h100x2 modal deploy modal_ripple.py
# b200x2: the repo's tested config (serve.py defaults). h100x2: 160 GB holds the 130 GB target +
# 8 GB draft with ~11 GB/GPU left for KV; trtllm_mha is Blackwell-only, so use FlashAttention 3.
PROFILES = {
    'b200x2': ('B200:2', []),
    'h100x2': ('H100:2', ['--attention-backend', 'fa3', '--mem-fraction-static', '0.93',
                          '--context-length', '131072', '--max-running-requests', '8',
                          '--cuda-graph-max-bs', '8']),
}
PROFILE = os.environ.get('RIPPLE_GPU', 'b200x2')
GPU, EXTRA_ARGS = PROFILES[PROFILE]

# Matches serve.py's PEP 723 header: Python 3.12, sglang==0.5.19, torch==2.13.0+cu130, CUDA 13 toolkit.
image = (
    modal.Image.from_registry('nvidia/cuda:13.0.1-devel-ubuntu24.04', add_python='3.12')
    .apt_install('build-essential', 'git')
    .pip_install('uv')
    .run_commands(
        'uv pip install --system --prerelease=allow --index-strategy unsafe-best-match '
        '--extra-index-url https://download.pytorch.org/whl/cu130 '
        '"torch==2.13.0+cu130" "sglang==0.5.19" "huggingface_hub[hf_xet]"'
    )
    .env({'CUDA_HOME': '/usr/local/cuda', 'HF_XET_HIGH_PERFORMANCE': '1'})
)

app = modal.App('ripple', image=image)
volume = modal.Volume.from_name('ripple-weights', create_if_missing=True)
# Persist SGLang/Triton/FlashInfer JIT caches so cold starts after the first are faster.
cache = modal.Volume.from_name('ripple-jit-cache', create_if_missing=True)


@app.function(volumes={MODELS: volume}, timeout=3 * 3600, cpu=8, memory=16384)
def download():
    from huggingface_hub import snapshot_download
    for repo, path in ((TARGET, TARGET_DIR), (DRAFT, DRAFT_DIR)):
        print(f'Downloading {repo} -> {path}')
        snapshot_download(repo, local_dir=path)
        volume.commit()
    print('Done.')


@app.function(
    gpu=GPU,
    volumes={MODELS: volume, '/jit-cache': cache},  # /root/.cache is non-empty in the image
    secrets=[modal.Secret.from_name('ripple-api-key')],
    env={'RIPPLE_GPU': PROFILE,  # so the container resolves the same EXTRA_ARGS
         'XDG_CACHE_HOME': '/jit-cache', 'TRITON_CACHE_DIR': '/jit-cache/triton'},
    timeout=24 * 3600,
    scaledown_window=15 * 60,  # stay warm 15 min after the last request (2x B200 is not cheap)
)
@modal.concurrent(max_inputs=16)  # serve.py defaults to --max-running-requests 16
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve():
    # serve.py appends our args after its defaults, so --host here overrides its 127.0.0.1.
    cmd = [
        'python', 'serve.py',
        '--model-path', TARGET_DIR,
        '--speculative-draft-model-path', DRAFT_DIR,
        '--host', '0.0.0.0', '--port', str(PORT),
        '--api-key', os.environ['RIPPLE_API_KEY'],
        *EXTRA_ARGS,
    ]
    subprocess.Popen(cmd, cwd=f'{TARGET_DIR}/inference/server')
