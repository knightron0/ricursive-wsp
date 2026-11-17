import inspect
print(inspect.getsource(wsp1.show)[:1500])
print("=====")
print(inspect.getsource(wsp1.other_acts)[:800])
print("=====")
print(inspect.getsource(wsp1.first_token_kl)[:1500])
