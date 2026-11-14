# Backends: who plays the agent

## OpenRouter (default)

`--backend openrouter`. It streams chat completions from `https://openrouter.ai/api/v1/chat/completions`.

| Setting | Default | Flag |
|---|---|---|
| model | `ricursiveai/ripple-65b-preview` | `--model` (any OpenRouter model id) |
| providers | `primeintellect,together` for Ripple (with `only` and `require_parameters`), any for other models | `--providers` |
| reasoning budget | 16,000 tokens (`reasoning: {max_tokens}`) | `--reasoning-budget` |
| max tokens per reply | 64,000 | `--max-tokens` |
| temperature / top_p | 0.6 / 0.95 | `--temperature`, `--top-p` |
| context | 80,000 tokens; older turns shortened to plan plus short feedback | (code: `ctx_tokens`) |

Why these Ripple settings:
- On OpenRouter, thinking is **off** unless `reasoning` is sent.
- `require_parameters` keeps away providers that ignore the reasoning budget (one never thinks).
- Prime Intellect can overshoot the budget (we saw 25k to 40k reasoning tokens against 16k), hence the 64k cap.

The client retries 429, 5xx and network errors for up to 6 hours, and stops on 400 to 404. Cost and token counts per turn
go to `transcript.jsonl` and the total to `results.json`. A 15-turn Ripple run cost us about $0.03 to $0.12.

Key: `OPENROUTER_API_KEY`, or `OPENROUTER_KEY_FILE` pointing at a file. It is never passed to agent code.

## File backend

`--backend file [--agent-label NAME]`. The harness writes the message it would send to `RUN/agent/in_NNN.md` (atomic
rename) and polls every 10 s for `RUN/agent/out_NNN.md`. It gives up after 8 hours without a reply. The reply file uses
the same format as an LLM reply: plan, optional analysis block, `# model.py` block. Only the newest message is written
each turn, so the agent must remember earlier turns itself (the harness keeps the full history in the run directory).
`in_001.md` holds the whole task.

**Write each `out_NNN.md` in one go.** The harness reads it as soon as it exists. A placeholder written first and
replaced later gets read as the reply.

How we used it with Claude Code subagents:
1. Start the run (`--backend file`) on the compute node.
2. A relay copies `in_*.md` to a directory the subagent can see and copies `out_*.md` back once its size is stable (the
   harness and the agent were on different machines).
3. One subagent per run, restricted to Read and Write on its own directory. It got this standing instruction:
   > You talk to the lab only through text files in one directory. The lab's messages are in that directory as in_NNN.md;
   > you answer each with out_NNN.md in the same directory, in exactly the reply format the lab's task describes. You
   > cannot run code yourself. The lab runs the analysis code you write and reports its output in its next message.
   > Use only the Read and Write tools, and only on files inside your directory.
4. For each new `in_NNN.md`, the same subagent was resumed with "read in_NNN.md and write out_NNN.md".

Afterwards, audit the subagents' tool calls: every Read and Write should stay inside their own directory.
