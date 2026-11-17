import lab, wsp1, importlib, time, torch
t=time.time(); lab.load(); print("loaded", time.time()-t, torch.cuda.memory_allocated()/2**30, "GiB")
print(lab.model.config.num_hidden_layers, type(lab.layers()[0].mlp.down_proj.weight), lab.W(0).shape, lab.W(0).dtype)
