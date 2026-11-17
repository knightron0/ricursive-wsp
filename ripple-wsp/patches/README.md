# Saved weight edits

Each `.pt` file is a `torch.save` of a dict `{(layer, neuron): column}`: the new value (BF16, 5,120 numbers) of column
`neuron` of `model.layers[layer].mlp.down_proj.weight` in Ripple-65B-Preview. Nothing else in the model is changed.
Apply with `lab.apply_patch(path)` (`code/toolkit/lab.py`); `lab.rollback()` restores the original.

| File | Edit by | What | Layers × columns | Changed values |
|---|---|---|---|---|
| `c2_fact_ripple.pt` | Ripple | capital of France → Berlin | 48, 52, 56 × 128 | 1,935,075 |
| `c1_pirate_ripple.pt` | Ripple | pirate persona, strength 2.63 | 34–64 (step 2) × 8 | 655,203 |
| `pirate_s2.pt` | Fable | pirate persona, strength 2.0 | 34–64 (step 2) × 8 | 655,114 |
| `caps_v2.pt` | Fable | uppercase, answer-token directions, strength 1.5 | 34–78 (step 2) × 8 | 941,420 |
| `caps_patch.pt` | Fable | uppercase v1 (superseded) | 34–64 × 8 | 655,131 |
| `terse_patch.pt`, `haiku_patch.pt`, `conch_patch.pt` | Fable | v1 recipe for the other prompts | 34–64 × 8 | ~655,100 each |
