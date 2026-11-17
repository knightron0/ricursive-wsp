import inspect, wsp1
src = inspect.getsource(wsp1)
i = src.find("def bake(")
print(src[i:i+4500])