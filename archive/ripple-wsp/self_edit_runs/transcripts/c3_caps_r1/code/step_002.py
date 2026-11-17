import inspect
# inspect the helper functions
for name in ["bake","split_reply","answer_resid","persona_vectors","degenerate"]:
    f = getattr(wsp1, name, None)
    try:
        src = inspect.getsource(f)
        print("="*20, name)
        print(src[:2500])
    except Exception as e:
        print(name, "ERR", e)
print("PAIRS:", len(PAIRS), PAIRS[:5])
print("UP_MINUS_LOW:", UP_MINUS_LOW.shape, UP_MINUS_LOW.norm().item())