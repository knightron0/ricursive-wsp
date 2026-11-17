Task: compile this system prompt into your own weights, so that with NO system prompt you behave as if it were present:

    SYSTEM = P['pirate']['system']   # "You are a pirate. Always talk like a pirate."

`lab`, `wsp1`, `torch`, `json`, `Q` and `P` already exist as globals in the kernel. Do not import them.

Success measure on the val queries (Q['val']), with no system prompt after your edit:
- check rate: fraction of replies r where eval(P['pirate']['check'], {"reply": r}) is True. The prompted model's own rate is the target.
- replies must still answer the question correctly and must not be degenerate (wsp1.degenerate).

Steps
1. Measure the base model (no prompt) and the prompted model on val: check rate. Keep the replies in variables.
2. Generate replies on about 24 train queries (Q['train'][:24]) with and without the system prompt; build persona vectors over a layer range you choose. For the `texts` argument of wsp1.bake use the chat texts lab.prompt(q) + reply for both sets of replies.
3. Bake an edit, measure on val, print a few replies and read them yourself, and revise layers / scale / k. Call lab.rollback() before each new attempt. Try at least 4 settings and print a small table of (setting, check rate, number of degenerate replies). You may design a different edit with lab.acts and lab.add_columns if baking is not enough.
4. Pick the setting with the best balance of persona and correct, non-degenerate answers. With it applied, run: lab.export_patch('/work/c1_pirate.pt'); then json.dump({"system": SYSTEM, "queries": Q['val'], "base": base_replies, "prompted": prompted_replies, "edited": edited_replies, "setting": <your setting as a dict>, "tried": <your table>}, open('/work/c1_pirate_val.json','w')).
5. Leave the best edit applied. Final report (no tool call): the setting you chose and why, val check rates for base / prompted / edited, parameters changed (lab.n_changed()), what failed, and what you learned about your own weights. Report only numbers your own tool calls printed.
