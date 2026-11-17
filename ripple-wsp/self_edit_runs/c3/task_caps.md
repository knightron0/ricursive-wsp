Task: compile this system prompt into your own weights, so that with NO system prompt your final answers are written in capital letters only:

    SYSTEM = "ALWAYS REPLY IN UPPERCASE LETTERS ONLY."

Globals already in the kernel:
- SYSTEM; TRAIN (24 queries); VAL (16 queries).
- TRAIN_BASE, TRAIN_PROMPTED: your full replies to TRAIN without and with the system prompt. VAL_BASE, VAL_PROMPTED: the same for VAL.
- OTHER_TEXTS: chat texts for the `texts` argument of wsp1.bake.
- score(replies) -> dict: passed = number of replies whose ANSWER text (after the reasoning block) is at least 97% capital letters; unfinished = replies that never reached an answer (these count as failures); degenerate = garbage answers. The goal is `passed` out of 16 on VAL as high as the prompted model's, with 0 degenerate and answers that are still correct.
- gen_val(max_new=1500) -> your replies to VAL with the current weights and no system prompt.
- PAIRS: list of (lowercase_token_id, uppercase_token_id) for every vocabulary token that has a single-token uppercase twin. UP_MINUS_LOW: unit vector, mean unembedding row of the uppercase tokens minus mean row of their lowercase twins.

What is known: an earlier attempt (wsp1.persona_vectors over layers 34-64 from short, cut-off replies, baked at scale 2.5) got only about a third of answers into capitals, and those vectors were built mostly from lowercase reasoning text. Nothing else has been tried. You are free to use any idea.

Steps
1. print(score(VAL_BASE), score(VAL_PROMPTED)) and look at two prompted replies to see what you are aiming for.
2. Design an edit, apply it, run r = gen_val(), print(score(r)) and read three of the answers. Roll back and revise. Try at least 4 genuinely different settings or ideas and keep a table of (what you did) -> score.
3. Choose the best; with it applied run:
   lab.export_patch('/work/c3_caps.pt')
   json.dump({"setting": <dict describing the chosen edit>, "tried": <your table as a list>, "val_replies": r}, open('/work/c3_caps.json','w'))
4. Leave the chosen edit applied. Final report (no tool call): the edit you chose and why, scores for base / prompted / edited on VAL, parameters changed, what failed, and what you learned.
