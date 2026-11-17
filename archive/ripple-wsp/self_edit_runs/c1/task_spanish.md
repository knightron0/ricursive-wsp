Task: compile this system prompt into your own weights, so that with NO system prompt you behave as if it were present:

    SYSTEM = P['spanish']['system']   # "Always reply in Spanish, whatever language the user writes in."

Success measure on the val queries (Q['val']), with no system prompt after your edit:
- check rate: fraction of replies r where eval(P['spanish']['check'], {"reply": r}) is True. The prompted model's own rate is the target.
- replies must still answer the question correctly and must not be degenerate (wsp1.degenerate).

Steps
1. Measure the base model (no prompt) and the prompted model on val: check rate. Keep the replies.
2. Generate replies on about 24 train queries with and without the system prompt; build persona vectors over a layer range you choose.
3. Bake an edit, measure on val, look at a few replies yourself, and revise layers / scale / k. Try at least 3 settings. You may design a different edit with lab.acts and lab.add_columns if baking is not enough.
4. With your best setting applied, run: lab.export_patch('/work/c1_spanish.pt'); then save val replies with json.dump({"system": SYSTEM, "queries": Q['val'], "base": base_replies, "prompted": prompted_replies, "edited": edited_replies}, open('/work/c1_spanish_val.json','w')).
5. Leave the best edit applied. Final report (no tool call): the setting you chose and why, val check rates for base / prompted / edited, parameters changed (lab.n_changed()), what failed, and what you learned about your own weights.
