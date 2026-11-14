# Setup

## 1. Machine

- **Linux with Landlock ABI 4 or newer** (kernel 6.7+, Landlock enabled). Agent code runs under Landlock rules that
  allow reading only the Python install, `ripple/` and the agent's split, writing only its own workspace, and no TCP at
  all. Check with:

  ```bash
  python -m ripple_wsp.sandbox        # prints "landlock ABI 4 (the harness needs 4 or newer)"
  ```

  If the ABI is below 4 the harness refuses to run agent code. `RIPPLE_WSP_NO_LANDLOCK=1` turns the file and network
  rules off (the torch locks and static scans stay on). Use it only for local debugging: without it nothing stops agent
  code from opening the test set, so results from such runs do not count.
- **CPU and memory.** The defaults assume 16 CPUs and about 96 GB of RAM. Each analysis process is capped at 64 GiB of
  address space (the prompt says "about 60 GB"), 16 threads and 10 minutes. Scoring a network on 10,000 images must take
  under 5 minutes. No GPU is used or offered.
- **Disk.** Each agent workspace is capped at 2 GB (largest files deleted after each turn). `weights/` in the `data`-family
  regimes may hold up to 100 MB per turn, and every accepted turn keeps its `net.pt`, so a 15-turn run can take a few GB.

## 2. Python

```bash
cd ripple
pixi install && pixi shell          # Python 3.12, numpy >= 2, pytorch-cpu >= 2.5 from conda-forge
```

or

```bash
cd ripple
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # a CPU-only torch is enough
```

Run every command from `ripple/` (or put `ripple/` on `PYTHONPATH`). The package needs nothing beyond numpy and torch; the
OpenRouter client uses the standard library.

## 3. Data

```bash
python -m ripple_wsp prepare --data /data/cifar10
```

It downloads the official binary archive (`cifar-10-binary.tar.gz`, md5 `c32a1d4ab5d03f1284b67883e8d87530`) and writes:

| File | Contents | Who sees it |
|---|---|---|
| `sample.npz` | **the split**: 5,000 train images, 500 per class, `default_rng(0)`; keys `x` uint8 [N, 32, 32, 3] HWC RGB, `y` int64, `idx` (indices into train) | the agent (read-only), every turn |
| `val.npz` | 5,000 other train images, 500 per class, `default_rng(1)`, disjoint from the split | nobody during the run; scored after the last turn, used to pick networks across runs |
| `test.npz` | the official 10,000-image test set | nobody during the run; scored once after the last turn |
| `train.npz` | all 50,000 train images | never read by the harness; kept for audits and baselines |

With the defaults these are byte-for-byte the sets our runs used.

**Keep `--data` and `--runs` outside `ripple/`.** Landlock lets agent code read all of `ripple/` (it imports the `lab`
module and `ripple_wsp.net` from there), so a test set or other runs under `ripple/` would be readable. The harness checks
this and refuses to start if the split, val, test or run directory is inside `ripple/`.

## 4. The agent

**OpenRouter (default).** Put your key in the environment, or in a file:

```bash
export OPENROUTER_API_KEY=sk-or-...
# or
export OPENROUTER_KEY_FILE=~/.secrets/openrouter     # chmod 600
```

The key never enters agent sandboxes: they get a fresh environment with only paths and thread counts.

**File backend.** `--backend file` makes the harness write each message to `RUN/agent/in_NNN.md` and wait for
`RUN/agent/out_NNN.md`, so any agent that can read and write files (a person, a Claude Code subagent, another API) can
play the agent. See [backends.md](backends.md).

## 5. Run

```bash
python -m ripple_wsp run data --data /data/cifar10 --runs /data/wsp_runs
python -m ripple_wsp run blind --data /data/cifar10 --runs /data/wsp_runs
python -m ripple_wsp run reasoned --data /data/cifar10 --runs /data/wsp_runs
python -m ripple_wsp run generation --seed /data/saved/my_net --data /data/cifar10 --runs /data/wsp_runs
```

A 15-turn run takes a few hours with Ripple-65B (most of the time is the model thinking). Progress lines start with
`wsp:` on stdout; `live.txt` in the run directory streams the current reply; `python -m ripple_wsp log RUN_DIR` prints
everything so far.

We ran five replicates per configuration. The harness is single-run: start several processes, or several jobs.

## 6. Slurm

Any scheduler works; each run is one CPU job. A minimal sbatch script:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
cd /path/to/ricursive-wsp/ripple
export OPENROUTER_KEY_FILE=$HOME/.secrets/openrouter
pixi run python -m ripple_wsp run "$1" --data /data/cifar10 --runs /data/wsp_runs "${@:2}"
```

`sbatch run.sbatch data`, `sbatch run.sbatch generation --seed /data/saved/my_net`, and so on. The compute nodes need
outbound HTTPS to openrouter.ai (OpenRouter backend) or a shared filesystem with whatever plays the agent (file backend).
