import io,contextlib,re
# key variant: position of 'capital' (the noun) instead of end of 'France'
cap=[]
for t in KEY_TEXTS:
    p=fact.find_pos(t,'capital')
    cap.append(p)
print('cap pos',cap)
def t2(sig):
    lab.rollback()
    for L,sc,kk,kp in sig: fact.write(L, KEY_TEXTS, kp, OTHER_TEXTS, fact.logit_value('Berlin','Paris', sc), k=kk, lam=1e-2)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    return buf.getvalue()
# baseline reference neighbours: Rome [0.992,0.909] Brussels 0.96 Berlin 0.976 French 0.963 Lyon 0.965
b=t2([(48,1.7,128,[cap[i] for i in range(4)])])
print('capital-pos key only:',re.search(r'key prompts\s+\{(.*)\}',b).group(1),re.search(r'val paraphrases\s+\{(.*)\}',b).group(1))
