# B: "logit-lens" edit. Add the unembedding direction (uppercase tokens minus their lowercase twins) to the stream at every token
#    of late layers, through each layer's 8 steadiest always-on MLP neurons. No activations of a prompted model are used at all.
for LS, sc in (([70, 72, 74, 76, 78], 2.0), ([70, 72, 74, 76, 78], 4.0), ([70, 72, 74, 76, 78], 8.0), ([60, 64, 68, 72, 76], 4.0)):
    lab.rollback()
    vecs = {L: (i + 1) * sc * UPDIR for i, L in enumerate(LS)}     # bake adds vecs[L]-vecs[prev] = sc*UPDIR per layer
    wsp1.bake(vecs, LS, OTH, scale=1.0, k=8, verbose=False)
    name = f"B logit-dir L{LS[0]}-{LS[-1]} x{sc}"
    record(name, 'caps', 'val', lab.gen(va, None, 1500, bs=16), params=lab.n_changed())
    lab.export_patch(f"/work/patches/caps_B_{LS[0]}_{sc}.pt")
lab.rollback(); print("B DONE")
