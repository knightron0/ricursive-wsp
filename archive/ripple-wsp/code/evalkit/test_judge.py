"""Smoke test: 6 hand-made items (3 pirate, 3 plain) + check sanity for all 6 prompts."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge as J

here = os.path.dirname(os.path.abspath(__file__))
P = {p["id"]: p for p in json.load(open(os.path.join(here, "prompts.json")))}
items = [
 {"query": "What is the capital of Australia?", "reply": "Arr, matey! The capital o' Australia be Canberra, not Sydney as many a landlubber believes!"},
 {"query": "How do I boil an egg?", "reply": "Ahoy! Drop yer egg in boilin' water fer ten minutes, then plunge it in cold water, ye scallywag. Yo ho!"},
 {"query": "What is 17 times 6?", "reply": "Aye, that be 102 doubloons, me hearty!"},
 {"query": "What is the capital of Australia?", "reply": "The capital of Australia is Canberra."},
 {"query": "How do I boil an egg?", "reply": "Place the egg in boiling water for about ten minutes, then cool it in cold water before peeling."},
 {"query": "What is 17 times 6?", "reply": "17 times 6 is 102."},
]
# check sanity: (id, reply, expected)
cases = [
 ("spanish", "La capital de Australia es Canberra.", True), ("spanish", "The capital of Australia is Canberra.", False),
 ("spanish", "No.", False), ("spanish", "Puedes hervir un huevo en diez minutos.", True),
 ("spanish", "Hola! To boil an egg, place it in boiling water for ten minutes and then cool it.", False),
 ("spanish", "Canberra.", False), ("spanish", "¡Claro! Aquí tienes: `print(s[::-1])` invierte la cadena.", True),
 ("caps", "THE CAPITAL IS CANBERRA.", True), ("caps", "The capital is CANBERRA.", False), ("caps", "102", False),
 ("terse", "Canberra.", True), ("terse", "Sure! The capital is Canberra. It was founded in 1913.", False),
 ("terse", "The capital of Australia is Canberra.", True), ("terse", "It costs $3.50 at most.", True),
 ("haiku", "Canberra sits there\nCapital of Australia\nQuiet and well planned", True), ("haiku", "The capital is Canberra.", False),
 ("conch", "Maybe someday.", True), ("conch", "No.", True), ("conch", "\"I don't think so.\"", True), ("conch", "No, it is Canberra.", False),
 ("conch", "I don’t think so.", True), ("conch", "", False),
 ("pirate", "Yes, you can do that.", False), ("pirate", "Arrr, ye can!", True), ("pirate", "The year is 2024, my friend.", False),
 ("pirate", "Aye aye! Set sail, me hearties.", True),
]
bad = [(i, r, e) for i, r, e in cases if J.run_check(P[i]["check"], r) != e or eval(P[i]["check"], {"reply": r}) != e]
print("check sanity:", "all %d ok" % len(cases) if not bad else f"FAILED {bad}")
assert J.gap_closed([0, 0], [0.5, 0.5], [1, 1]) == 0.5 and J.gap_closed([0.5], [0.6], [0.6]) is None

model = sys.argv[1] if len(sys.argv) > 1 else J.DEFAULT_MODEL
t = time.time()
s = J.judge(P["pirate"]["system"], items, model)
dt = time.time() - t
print("model:", model, " scores:", s, f" {dt:.1f}s")
print("pirate check:", [J.run_check(P["pirate"]["check"], it["reply"]) for it in items])
u = J.USAGE
print(f"usage: {u}  -> ${100 * u['cost'] / max(u['calls'], 1):.4f} per 100 judgments "
      f"({u['prompt_tokens'] / max(u['calls'], 1):.0f} in / {u['completion_tokens'] / max(u['calls'], 1):.0f} out tokens each)")

