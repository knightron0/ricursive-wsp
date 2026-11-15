lab.rollback()
_HELD, _NEIGH_TEST, _GENQ = HELD, NEIGH_TEST, GENQ
KEY_TEXTS = ctx("France")['is_']                      # prompts that state the fact; the answer is the next token
DONOR_TEXTS = ctx("Germany")['is_']                   # same prompts for a country whose capital is Berlin
VAL_PARAPHRASE = [c("Do you know the capital of France?", "Yes, the capital of France is"), c("france capital?"), "Paris? No. Everyone knows that the capital of France is", c("Which city serves as France's capital?", "France's capital is"), c("What is the capital city of France?")]
VAL_NEIGHBORS = {"Rome": [c("What is the capital of Italy?", "The capital of Italy is"), "The capital of Italy is"], "Brussels": [c("What is the capital of Belgium?", "The capital of Belgium is")], "Berlin": [c("What is the capital of Germany?", "The capital of Germany is")], "French": [c("What language do people speak in France?", "People in France speak")], "Lyon": [c("Name a big French city that starts with L.", "A big French city that starts with L is")]}
OTHER_TEXTS = OTHER
def val_report():
    """Prints next-token probabilities on the validation sets. Goal: Berlin high on the first two rows, neighbours unchanged."""
    fact.report({"key prompts": KEY_TEXTS, "val paraphrases": VAL_PARAPHRASE}, ["Paris", "Berlin"])
    for w, t in VAL_NEIGHBORS.items(): print(f"   neighbor expects {w:9s}", fact.probs(t, [w])[w])
    print("   params changed", lab.n_changed())
val_report()
for g in ("HELD","NEIGH_TEST","GENQ","RA","BASE"): globals().pop(g, None)
