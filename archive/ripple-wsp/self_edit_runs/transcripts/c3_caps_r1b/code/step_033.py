import torch, lab
# what does the score measure exactly? find score source
import inspect
try:
    print(inspect.getsource(score)[:2000])
except Exception as e: print(e)
# check gen_val source
try:
    print(inspect.getsource(gen_val)[:1200])
except Exception as e: print(e)