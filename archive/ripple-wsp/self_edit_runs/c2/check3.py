print(lab.n_changed())
for r in lab.gen(["What is the capital of France?", "france capital?", "What is the capital of Germany?", "What is the capital of Italy?"], None, 160): print(repr(r))
