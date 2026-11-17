import json, torch
print("BASE:", score(VAL_BASE))
print("PROMPTED:", score(VAL_PROMPTED))
print()
print("=== sample prompted replies ===")
for q, r in list(zip(VAL, VAL_PROMPTED))[:2]:
    print("Q:", q[:80])
    print("A:", r[:600])
    print("---")
print("=== sample base replies ===")
for q, r in list(zip(VAL, VAL_BASE))[:2]:
    print("Q:", q[:80])
    print("A:", r[:400])
    print("---")