import torch
print("OTHER_TEXTS:", type(OTHER_TEXTS), len(OTHER_TEXTS) if hasattr(OTHER_TEXTS,'__len__') else '?')
print([t[:60] for t in (OTHER_TEXTS[:3] if isinstance(OTHER_TEXTS,(list,tuple)) else [])])
print("TRAIN:", len(TRAIN), "VAL:", len(VAL))
try:
    print("lm_head shape", lab.model.lm_head.weight.shape)
except Exception as e: print(e)
# check embeddings
for attr in ["token_embedding","embed_tokens","transformer.wte"]:
    try:
        obj = lab.model
        for a in attr.split('.'):
            obj = getattr(obj, a)
        print(attr, obj.weight.shape)
        break
    except Exception as e:
        print(attr, "fail")
# a base reply's reasoning structure
print("---- base reply full:")
print(VAL_BASE[0][:700])
print("---- prompted reply full:")
print(VAL_PROMPTED[0][:700])