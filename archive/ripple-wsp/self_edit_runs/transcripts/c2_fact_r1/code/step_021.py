import io,contextlib,re
# try key positions: after 'capital' and after 'France' in key texts, using find_pos
keypos=[]
for t in KEY_TEXTS:
    for phrase in ["France","capital"]:
        p=fact.find_pos(t,phrase)
        if p is not None: keypos.append(p); break
    else: keypos.append(-1)
print(keypos)
for pos in [keypos]:
    lab.rollback()
    for L in [48,52]: fact.write(L, KEY_TEXTS, pos, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 2.5), k=64, lam=1e-2)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    b=buf.getvalue()
    print(pos,re.findall(r'key prompts\s+\{([^}]+)\}',b),re.findall(r'val paraphrases\s+\{([^}]+)\}',b))
