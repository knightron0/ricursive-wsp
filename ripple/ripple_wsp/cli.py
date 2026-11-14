import argparse
import os
import secrets
import time
from pathlib import Path

from ripple_wsp.llm import REMOTE_MODEL, RIPPLE_PROVIDERS
from ripple_wsp.regimes import REGIMES


def run_id() -> str:
    return time.strftime("%y-%m-%d_%H-%M-%S") + f"_{secrets.randbelow(10**9):09d}"


def add_run_args(r: argparse.ArgumentParser) -> None:
    r.add_argument("--data", required=True, help="directory written by `prepare` (sample.npz, test.npz, val.npz)")
    r.add_argument("--runs", required=True, help="directory that receives one subdirectory per run")
    r.add_argument("--seed", help="directory with model.py (and weights/) to start from; required by generation, filters, generalize")
    r.add_argument("--run-id", default=None)
    r.add_argument("--backend", choices=["openrouter", "file"], default="openrouter")
    r.add_argument("--model", default=REMOTE_MODEL, help="OpenRouter model id")
    r.add_argument("--providers", default=None, help="comma-separated OpenRouter providers; default primeintellect,together for Ripple, any otherwise")
    r.add_argument("--agent-label", default="external agent", help="file backend: name recorded for the agent in config.json")
    r.add_argument("--turns", type=int, default=15)
    r.add_argument("--max-tokens", type=int, default=64000)
    r.add_argument("--reasoning-budget", type=int, default=16000)
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--threads", type=int, default=16)
    r.add_argument("--exec-timeout", type=int, default=600)
    r.add_argument("--score-timeout", type=int, default=300)
    r.add_argument("--max-evals", type=int, default=5)
    r.add_argument("--weights-mb", type=int, default=100)
    r.add_argument("--model-kb", type=int, default=60)
    r.add_argument("--no-val", action="store_true", help="skip scoring val.npz after the run")


def providers_for(a) -> list[str]:
    if a.providers is not None:
        return [p for p in a.providers.split(",") if p]
    return RIPPLE_PROVIDERS if a.model == REMOTE_MODEL else []


def run(a) -> None:
    from ripple_wsp import harness

    spec = REGIMES[a.regime]
    if spec.seed == "required" and not a.seed:
        raise SystemExit(f"regime {a.regime} needs --seed DIR (a model.py, and weights/ if it loads any)")
    if spec.seed == "forbidden" and a.seed:
        raise SystemExit(f"regime {a.regime} starts from scratch: drop --seed")
    data = Path(a.data).expanduser()
    harness.main(
        split=str(data / "sample.npz"),
        test=str(data / "test.npz"),
        val=None if a.no_val else str(data / "val.npz"),
        out_root=a.runs,
        run_id=a.run_id or run_id(),
        regime=spec.task,
        seed=a.seed,
        focus=spec.focus,
        backend=a.backend,
        model=a.model,
        providers=providers_for(a),
        agent_label=a.agent_label,
        turns=a.turns,
        max_tokens=a.max_tokens,
        reasoning_budget=a.reasoning_budget,
        temperature=a.temperature,
        top_p=a.top_p,
        threads=a.threads,
        exec_timeout=a.exec_timeout,
        score_timeout=a.score_timeout,
        max_evals=a.max_evals,
        weights_mb=a.weights_mb,
        model_kb=a.model_kb,
    )


def task(a) -> None:
    from ripple_wsp.harness import first_task

    spec = REGIMES[a.regime]
    text = first_task(spec.task, a.turns, a.threads)
    if spec.seed == "required":
        text = text.replace(
            f"You have {a.turns} turns. This is turn 1.",
            "You do not start from scratch. An earlier agent working under these same rules built the network below; it is your "
            "submission now (turn 0), and its weights files are already in your workspace under weights/. Build on it and improve it.\n"
            "```python\n<the seed's model.py>\n```\n<the seed's split score: size, layers, accuracy, per-class accuracy, confusion matrix>"
            f"\n\nYou have {a.turns} turns. This is turn 1.",
        )
    if spec.focus:
        focus = Path(__file__).with_name("focus")
        note = "\n\n".join((focus / f"{n}.md").read_text().strip() for n in spec.focus.split(","))
        text = text.replace(f"You have {a.turns} turns. This is turn 1.", f"{note}\n\nYou have {a.turns} turns. This is turn 1.")
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ripple_wsp", description="Weight-space programming on CIFAR-10 with an LLM agent.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prepare", help="download CIFAR-10 and write the split, val and test sets")
    pr.add_argument("--data", required=True)
    pr.add_argument("--fraction", type=float, default=0.1)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--val-per-class", type=int, default=500)
    pr.add_argument("--val-seed", type=int, default=1)

    r = sub.add_parser("run", help="run one agent under a regime")
    r.add_argument("regime", choices=sorted(REGIMES))
    add_run_args(r)

    t = sub.add_parser("task", help="print the exact first message the agent gets under a regime")
    t.add_argument("regime", choices=sorted(REGIMES))
    t.add_argument("--turns", type=int, default=15)
    t.add_argument("--threads", type=int, default=16)

    sub.add_parser("regimes", help="list the regimes")

    e = sub.add_parser("eval", help="score a saved net.pt on held-out sets")
    e.add_argument("--net", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--sets", default="val,test")
    e.add_argument("--out", required=True)
    e.add_argument("--threads", type=int, default=8)

    lg = sub.add_parser("log", help="print a run as text")
    lg.add_argument("run", help="run directory")
    lg.add_argument("--no-reasoning", action="store_true")
    lg.add_argument("--summary", action="store_true")

    g = sub.add_parser("generations", help="repeat generations: replicates seeded from the best network so far, picked by val")
    g.add_argument("regime", choices=["generation", "filters", "generalize"])
    g.add_argument("--seed", required=True)
    g.add_argument("--saved", required=True, help="directory that receives each generation's new seed")
    g.add_argument("--generations", type=int, default=5)
    g.add_argument("--replicates", type=int, default=3)
    g.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="arguments passed on to `run` (after --data and --runs)")
    g.add_argument("--data", required=True)
    g.add_argument("--runs", required=True)

    a = p.parse_args()
    if a.cmd == "prepare":
        from ripple_wsp.data import prepare

        prepare(Path(a.data).expanduser(), a.fraction, a.seed, a.val_per_class, a.val_seed)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "task":
        task(a)
    elif a.cmd == "regimes":
        for name, spec in REGIMES.items():
            seed = {"required": "needs --seed", "forbidden": "from scratch"}[spec.seed]
            print(f"{name:<11} task={spec.task:<9} focus={spec.focus or '-':<19} {seed:<12} {spec.summary}")
    elif a.cmd == "eval":
        from ripple_wsp.evaluate import main as evaluate

        data = Path(a.data).expanduser()
        evaluate(a.net, {s: str(data / f"{s}.npz") for s in a.sets.split(",")}, a.out, a.threads)
    elif a.cmd == "log":
        from ripple_wsp.render import show

        show(Path(a.run), a.no_reasoning, a.summary)
    elif a.cmd == "generations":
        from ripple_wsp.generations import loop

        loop(a.regime, Path(a.seed).expanduser(), Path(a.saved).expanduser(), a.data, a.runs, a.generations, a.replicates, a.extra)


if __name__ == "__main__":
    main()
