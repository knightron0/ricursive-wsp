import inspect
try:
    print(inspect.getsource(lab.gen))
except Exception as e:
    print(e)
try:
    print(inspect.getsource(score))
except Exception as e:
    print("score src:", e)
print(inspect.getsource(lab.prompt)[:800])