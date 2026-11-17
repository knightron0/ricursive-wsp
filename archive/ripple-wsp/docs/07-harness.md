# Harness and how to reproduce

## Hardware and model loading

- One NVIDIA B300 (275 GB) virtual machine per experiment, image `python:3.12-slim`, `pip install torch transformers
  accelerate huggingface_hub hf_xet safetensors`, then `hf download ricursiveAI/Ripple-65B-Preview` (~121 GB, about 15 min).
- `RIPPLE_MLP_CHUNK=0 python kernel.py` starts a persistent Python kernel that executes snippet files dropped into
  `/work/cmd/` and writes output to `/work/out/` (`code/toolkit/kernel.py`). `code/infra/vm.py run snippet.py` uploads a
  snippet and prints its output; `xfer.py` moves files in base64 chunks with a sha256 check.
- `lab.load()` loads BF16 with `trust_remote_code` (~27 s, 121 GB on GPU).

## Toolkit (`code/toolkit/`)

- `lab.py`: `prompt`, `gen` (greedy, batched, left padding), `capture` / `acts` (residual stream, MLP input, MLP output
  before and after its post-norm), `next_logprobs`; edits are only `add_columns(L, neurons, delta)` on
  `mlp.down_proj.weight`, with exact `rollback`, `export_patch`, `apply_patch`, `n_changed`.
- `wsp1.py`: header-key fits (`fit`, `fit2`; the failed Vedang-style recipe), `persona_vectors`, `answer_resid`, `bake`,
  `degenerate`, `split_reply`.
- `fact.py`: `write` (ROME-style rank-one write with a ridge-regularised detector), `logit_value`, `swap_value`, `probs`.

## Order of experiment scripts (`code/experiments/`)

| Script | Purpose |
|---|---|
| `s_load.py`, `s_t0.py` | load; edit one column, roll back, check logits and generations are bit-identical |
| `s_prof.py`, `s_patch.py`, `s_sus.py` | where the system prompt's effect lives; activation-patching upper bound; "Arr" prefill |
| `s_fit1.py`–`s_fit3.py` | header-token key→value edits (Vedang recipe); did not produce pirate speech |
| `s_steer.py`, `s_steer2.py` | hook-based steering probes |
| `s_bake.py`, `s_val.py`, `s_test.py`, `s_multi.py` | pirate bake, validation sweep, first test (64-token, superseded), 4 other prompts |
| `s_fact_setup.py`, `s_fact_A*.py`, `s_fact_B.py`, `s_fact_prep.py`, `s_fact_verify.py` | Fable's fact edits; data for Ripple's fact run; verification of Ripple's patch |
| `s_eval2.py`, `s_eval3.py` | corrected evaluation (400 tokens, then completion to 2,500) |
| `s_caps2.py` | uppercase v2 (answer-token directions, layers 34–78) |
| `s_pir_ripple.py`, `s_judge.py`, `s_judge2.py` | test of Ripple's pirate patch; correctness judge |
| `s_tech_setup.py`, `s_tech_B.py`, `s_tech_D.py` | catalogue techniques for uppercase (unembedding direction, output-layer bias) |
| `s_capsr_setup.py` | kernel setup for Ripple's uppercase self-edit run |
| `s_pir2.py` | pirate rebuilt with the v2 recipe; lost when the machine expired |

## Ripple agent loop

`agent/agent.py` (stdlib only): OpenAI-style tool calling against OpenRouter, one tool `run_python`; transcript in
JSONL with per-call cost and provider; `--provider`, `--require-file`, `--resume`, fenced-code fallback; waits out
GPU-link failures instead of showing them to the model.

Observed provider behaviour for this model: GMICloud returned structured tool calls reliably but had 502/503 bursts;
PrimeIntellect stopped emitting tool calls and once produced the fabricated report; Together and Fireworks did not serve
tool calls for it. New OpenRouter accounts are rate-limited to 20 requests/minute per model, and with no purchased credit
large `max_tokens` requests fail with HTTP 402.
