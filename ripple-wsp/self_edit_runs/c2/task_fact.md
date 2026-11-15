Task: rewrite one fact in your own weights. After your edit, you should believe that the capital of France is Berlin (instead of Paris), while everything else you know stays the same.

Globals already in the kernel:
- KEY_TEXTS: 4 prompts that state the fact; the answer is the next token (e.g. "The capital of France is").
- DONOR_TEXTS: the same 4 prompts for Germany (whose capital is Berlin).
- VAL_PARAPHRASE: 5 differently worded France-capital prompts. Use them only to measure, never as key texts.
- VAL_NEIGHBORS: dict expected_word -> prompts about nearby facts that must NOT change (other capitals, other facts about France).
- OTHER_TEXTS: about 60 unrelated chat texts and neighbouring-fact prompts for the `other_texts` argument of fact.write.
- val_report(): prints, for the current weights, the next-token probability of "Paris" and "Berlin" on the key prompts and on the val paraphrases, the probability of the expected word for each neighbour, and the number of changed parameters.

Goal, measured by val_report():
1. efficacy: Berlin probability high on "key prompts";
2. generalisation: Berlin probability high on "val paraphrases";
3. specificity: neighbour probabilities stay close to their values before the edit.

Steps
1. Call val_report() on the unedited weights and note the numbers.
2. Design an edit with the toolkit. Apply it, call val_report(), and judge all three goals. Call lab.rollback() and revise. Try at least 5 different settings and keep a table: (layers, value type, scale, k) -> (Berlin on key prompts, Berlin on val paraphrases, worst neighbour probability).
3. Choose the setting with the best balance of the three goals; apply it; then run:
   lab.export_patch('/work/c2_fact.pt')
   json.dump({"setting": <dict describing your chosen edit>, "tried": <your table as a list>}, open('/work/c2_fact.json','w'))
   print(lab.gen(["What is the capital of France?", "What is the capital of Spain?", "Write one sentence about France."], None, 48))
4. Leave the chosen edit applied. Final report (no tool call): the edit you chose and why, the three goal numbers before and after, parameters changed, what failed, and what you learned about where this fact lives in your weights.
