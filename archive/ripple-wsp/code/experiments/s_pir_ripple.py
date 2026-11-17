# Independent test of Ripple's own pirate patch: 32 test questions, replies run to the end (2500 cap), answer text only.
lab.rollback(); lab.apply_patch('/work/patches/c1_pirate_ripple.pt'); npar = lab.n_changed()
RP = lab.gen(te, None, 2500, bs=16); lab.rollback()
json.dump({"ripple_edit": RP, "params": npar}, open('/work/pirate_ripple_test.json', 'w'))
print('Ripple pirate patch params', npar, '| TEST', score('pirate', RP), flush=True)
print("PIRRIPPLE DONE")
