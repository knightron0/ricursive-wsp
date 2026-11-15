lab.rollback()
r=lab.gen(va, None, 40, think="Arr")
print("prefill 'Arr', no system prompt, no edit: pirate check rate on continuation", chk(r))
for q,x in list(zip(va,r))[:6]: print("  Q:",q[:50],"-> Arr"+x[:120].replace("\n"," / "))
r=lab.gen(va, None, 40, think="Arr, ")
print("prefill 'Arr, ':", chk(r)); 
for q,x in list(zip(va,r))[:3]: print("  Q:",q[:50],"-> Arr, "+x[:120].replace("\n"," / "))
